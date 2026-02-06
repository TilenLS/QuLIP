from lambeq import PytorchQuantumModel
from tqdm import tqdm
import torch
import torch.nn.functional as F
from opt_einsum import contract_expression
from collections import defaultdict
from cotengra import einsum
from util import *

class EinsumModel(PytorchQuantumModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_temp = torch.nn.Parameter(torch.tensor([0.1]))
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sym2weight = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = []
        self.path_cache = {}
        self.compiled = False

    def from_einsum(self, einsum_arr):
        self.symbols = []
        for _, tensors, scalar in tqdm(einsum_arr):
            for symbol, rot_type in tensors:
                if rot_type != 0 and symbol not in self.symbols:
                    self.symbols.append(symbol)

    def initialise_weights(self) -> None:
        self._reinitialise_modules()
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`PytorchQuantumModel.from_diagrams()`.')
        self.weights = torch.nn.Parameter(torch.empty(len(self.symbols)).uniform_(0, 2*torch.pi), requires_grad=True)
        # self.weights = torch.nn.Parameter(torch.empty(len(self.symbols)).uniform_(-0.01, 0.01), requires_grad=True)
        self.sym2weight = {sym: idx for sym, idx in zip(self.symbols, range(len(self.weights)))}

    def compile_paths(self, einsum_arr):
        self.compiled = True
        for tnet in tqdm(einsum_arr):
            einsum_string, tensors, scalar = tnet
            shapes = []
            constants = []
            for i, (tensor, rot_type) in enumerate(tensors):
                if rot_type == 0:
                    shapes.append(tensor)
                    constants.append(i)
                else:
                    if rot_type == 3:
                        shapes.append((2, 2, 2, 2))
                    else:
                        shapes.append((2, 2))
            cache_key = (einsum_string, tuple(shapes))
            if cache_key not in self.path_cache: self.path_cache[cache_key] = contract_expression(einsum_string, *shapes, constants=constants)

    def contract(self, circ):
        einsum_string, tensors, scalar = circ
        param_arr = [data if (rot_type==0) else self.map_phase(self.weights[self.sym2weight[data]], rot_type) for data, rot_type in tensors]
        if self.compiled:
            return self.path_cache[(einsum_string, tuple([t.shape for t in param_arr]))](*param_arr) * scalar
        return einsum(einsum_string, *param_arr)*scalar

    def batch_contract(self, crc_arr):
        return torch.stack([self.contract(circ) for circ in crc_arr])
    
    def map_phase(self, theta, rot_type=0):
        if rot_type == 1:
            return Rz(theta)
        elif rot_type == 2:
            return Rx(theta)
        elif rot_type == 3:
            return CRz(theta)

    def compile_batch(self, batch_recipes):
        groups = defaultdict(list)
        for i, (einsum_str, tensors, scalar) in enumerate(batch_recipes):
            groups[einsum_str].append((i, tensors, scalar))
        batch_size = len(batch_recipes)
        for einsum_str, items in groups.items():
            indices, tensor_lists, _ = zip(*items)
            minibatch_size = len(indices)
            gate_columns = zip(*tensor_lists)
            constants = []
            shapes = []
            inputs = []
            for i, column in enumerate(gate_columns):
                first_data, rot_type = column[0]
                if rot_type == 0:
                    expanded = first_data.unsqueeze(0).expand(minibatch_size, *first_data.shape)
                    shapes.append(expanded.shape)
                    inputs.append(expanded)
                    constants.append(i)
                elif rot_type == 1 or rot_type == 2:
                    shapes.append((minibatch_size, 2, 2))
                    inputs.append((minibatch_size, 2, 2))
                elif rot_type == 3:
                    shapes.append((minibatch_size, 2, 2, 2, 2))
                    inputs.append((minibatch_size, 2, 2, 2, 2))

            input_subscripts, output_subscript = einsum_str.split('->')
            batched_inputs = ["$" + s for s in input_subscripts.split(',')]
            batched_str = f"{','.join(batched_inputs)}->${output_subscript}"
            cache_key = (batched_str, tuple(shapes))
            if batched_str not in self.path_cache: self.path_cache[cache_key] = contract_expression(batched_str, *inputs, constants=constants)
    
    def compile_dataset(self, data_loader):
        for batch in data_loader:
            tn_batch, _, _ = batch
            self.compile_batch(tn_batch)


    def fast_batch_contract(self, batch_recipes):
        groups = defaultdict(list)
        for i, (einsum_str, tensors, scalar) in enumerate(batch_recipes):
            groups[einsum_str].append((i, tensors, scalar))

        batch_size = len(batch_recipes)
        results = torch.zeros(batch_size, *[2]*9, dtype=torch.complex128)
        for einsum_str, items in groups.items():
            indices, tensor_lists, scalars = zip(*items)
            minibatch_size = len(indices)
            gate_columns = zip(*tensor_lists)
            stacked_tensors = []
            shapes = []
            for column in gate_columns:
                first_data, rot_type = column[0]
                if rot_type != 0:
                    symbols = [data for data, rtype in column]
                    weight_indices = [self.sym2weight[s] for s in symbols]
                    thetas = self.weights[weight_indices]
                    if rot_type == 1: gate_batch = BatchRz(thetas)
                    elif rot_type == 2: gate_batch = BatchRx(thetas)
                    elif rot_type == 3: gate_batch = BatchCRz(thetas)
                    shapes.append(gate_batch.shape)
                    stacked_tensors.append(gate_batch)
                else:
                    shapes.append((minibatch_size, *first_data.shape))

            input_subscripts, output_subscript = einsum_str.split('->')
            batched_inputs = ["$" + s for s in input_subscripts.split(',')]
            batched_str = f"{','.join(batched_inputs)}->${output_subscript}"
            cache_key = (batched_str, tuple(shapes))

            view_shape = [len(scalars)] + [1] * 9
            scalar_tensor = torch.tensor(scalars).view(*view_shape)
            if cache_key not in self.path_cache:
                group_out = torch.einsum(batched_str, *stacked_tensors) * scalar_tensor
                self.path_cache[cache_key] = contract_expression(batched_str, *shapes)
            else:
                group_out = self.path_cache[cache_key](*stacked_tensors) * scalar_tensor

            results[list(indices)] = group_out
        return results

    # def forward(self, out_s, out_i):
    #     out_i = torch.stack([img_enc for img_enc in out_i]).to(self.device)
        #out_s = self.batch_contract(out_s)
        # out_s = self.fast_batch_contract(out_s)
        # out_i = out_i.reshape(out_i.size(0), -1)
        # out_s = out_s.reshape(out_s.size(0), -1)
        # out_i = F.normalize(out_i, dim=1, p=2)
        # out_s = F.normalize(out_s, dim=1, p=2)

        z = out_s @ out_i.conj().t()
        # mod = z.abs()
        # arg = torch.cos(torch.angle(z))
        # logits = mod*arg / torch.exp(self.log_temp)

        # Fubini-Study Metric
        # logits = (torch.pi - torch.acos(z.abs().clamp(0,1))) / torch.exp(self.log_temp)

        # labels = torch.arange(len(logits), device=self.device)
        # loss_s = self.loss_fn(logits, labels)
        # loss_i = self.loss_fn(logits.t(), labels)
        # return (loss_s + loss_i) / 2

        # return self.loss_fn(logits, torch.arange(len(logits)))


class QInfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.07, reduction='mean'):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.temperature = temperature
    
    def forward(self, txt, img):
        txt = F.normalize(txt.reshape(txt.size(0), -1), dim=1, p=2)
        img = F.normalize(img.reshape(txt.size(0), -1), dim=1, p=2)
        z = txt @ img.conj().t()
        logits = (torch.pi - torch.acos(z.abs().clamp(0,1))) / self.temperature
        labels = torch.arange(len(logits))
        loss_txt = self.cross_entropy(logits, labels)
        loss_img = self.cross_entropy(logits.t(), labels)
        return (loss_txt + loss_img) / 2        

def update_model(model, data_loader, loss_fn, acc_fn, optimizer):
    cum_loss = cum_acc = grad_norm = 0
    model.train()
    for batch in data_loader:
        circ, pos_img, neg_img = batch

        pos_img = torch.stack([img_enc for img_enc in pos_img])
        neg_img = torch.stack([img_enc for img_enc in neg_img])
        txt = model.fast_batch_contract(circ)

        pos_img = pos_img.reshape(pos_img.size(0), -1)
        neg_img = neg_img.reshape(neg_img.size(0), -1)
        txt = txt.reshape(txt.size(0), -1)

        optimizer.zero_grad()
        loss = loss_fn(txt, pos_img)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # model.log_temp.clamp(min=torch.exp(torch.tensor(0.1))
        optimizer.step()
        cum_loss += loss

        posres = acc_fn(pos_img, txt)
        negres = acc_fn(neg_img, txt)
        cum_acc += (torch.sum((posres > negres))/len(posres)).item()

        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        grad_norm += total_norm ** 0.5

    N = len(data_loader)
    return (cum_loss/N, cum_acc/N, grad_norm/N)
    
def eval_model(model, data_loader, acc_fn):
    cum_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            circ, pos_img, neg_img = batch

            pos_img = torch.stack([img_enc for img_enc in pos_img])
            neg_img = torch.stack([img_enc for img_enc in neg_img])
            txt = model.fast_batch_contract(circ)

            pos_img = pos_img.reshape(pos_img.size(0), -1)
            neg_img = neg_img.reshape(neg_img.size(0), -1)
            txt = txt.reshape(txt.size(0), -1)

            posres = acc_fn(pos_img, txt)
            negres = acc_fn(neg_img, txt)
            cum_acc += (torch.sum((posres > negres))/len(posres)).item()
    
    return cum_acc/len(data_loader)