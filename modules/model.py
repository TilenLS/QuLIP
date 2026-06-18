from lambeq import PytorchQuantumModel
from tqdm import tqdm
import torch
import torch.nn.functional as F
from opt_einsum import contract_expression
from collections import defaultdict
from cotengra import einsum
from modules.quantum import *

class EinsumModel(PytorchQuantumModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_temp = None 
        self.weights = None
        self.sym2weight = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path_cache = {}
        self.drop_pr = 0.0
        self.precision = torch.complex64
        self.qout = 9

    @property
    def temp(self):
        return torch.exp(self.log_temp)
    
    @property 
    def params(self):
        return self.weights
    
    @property
    def grad_norm(self):
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def from_einsum(self, einsum_arr):
        symbols_hash = set()
        self.symbols = []
        for _, tensors in tqdm(einsum_arr):
            for symbol, op_type in tensors:
                if (symbol is not None) and (symbol not in symbols_hash):
                    symbols_hash.add(symbol)
                    self.symbols.append(symbol)

    def initialise_weights(self, near_id=False, temp=0.07) -> None:
        self._reinitialise_modules()
        if not self.symbols:
            raise ValueError('Symbols not initialised. Instantiate through '
                             '`PytorchQuantumModel.from_diagrams()`.')

        self.log_temp = torch.log(torch.tensor(temp))

        if near_id:
            self.weights = torch.nn.Parameter(torch.empty(len(self.symbols)).uniform_(-0.01, 0.01), requires_grad=True)
        else:
            self.weights = torch.nn.Parameter(torch.empty(len(self.symbols)).uniform_(-torch.pi/2, torch.pi/2), requires_grad=True)
        self.sym2weight = {sym: idx for sym, idx in zip(self.symbols, range(len(self.weights)))}

    def compile_batch(self, batch_recipes):
        groups = defaultdict(list)
        for i, (einsum_str, tensors) in enumerate(batch_recipes):
            groups[einsum_str].append((i, tensors))

        for einsum_str, items in groups.items():
            indices, tensor_lists = zip(*items)
            minibatch_size = len(indices)
            gate_columns = zip(*tensor_lists)
            constants = []
            shapes = []
            inputs = []
            for i, column in enumerate(gate_columns):
                symbol, op_type = column[0]
                if symbol is None:
                    if op_type == 'sqrt':
                        data = torch.tensor(2.0**0.5, dtype=self.precision)
                    elif op_type == '0':
                        data = torch.tensor([1,0], dtype=self.precision)
                    elif op_type == 'H':
                        data = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=self.precision) / (2.0**0.5)
                    elif op_type == 'CX':
                        data = torch.block_diag(torch.eye(2), torch.tensor([[0.0,1.0],[1.0,0.0]], dtype=self.precision)).reshape(2,2,2,2)
                    data = data.unsqueeze(0).expand(minibatch_size, *data.shape)
                    shapes.append(data.shape)
                    inputs.append(data)
                    constants.append(i)
                else:
                    if op_type == 'Rz' or op_type == 'Rx' or op_type == 'Ry':
                        shape = torch.Size([minibatch_size, 2, 2])
                        shapes.append(shape)
                        inputs.append(shape)
                    elif op_type == 'CRz' or op_type == 'CRx' or op_type == 'CRy': 
                        shape = torch.Size([minibatch_size, 2, 2, 2, 2])
                        shapes.append(shape)
                        inputs.append(shape)

            input_subscripts, output_subscript = einsum_str.split('->')
            batched_inputs = ["$" + s for s in input_subscripts.split(',')]
            batched_str = f"{','.join(batched_inputs)}->${output_subscript}"
            cache_key = (batched_str, tuple(shapes))
            if cache_key not in self.path_cache: self.path_cache[cache_key] = contract_expression(batched_str, *inputs, constants=constants)
    
    def compile_dataset(self, data_loader, dataset='svo'):
        if dataset == 'svo':
            for batch in data_loader:
                tn_batch, _, _ = batch
                self.compile_batch(tn_batch)
        if dataset == 'aro':
            for batch in data_loader:
                pos_tn_batch, neg_tn_batch, _ = batch
                self.compile_batch(pos_tn_batch)
                self.compile_batch(neg_tn_batch)

    def compile_dataset2(self, data_loader, dataset='svo'):
        if dataset == 'svo':
            for batch in data_loader:
                txt, pos_img, neg_img = batch
                self.compile_batch(txt)
                self.compile_batch(pos_img)
                self.compile_batch(neg_img)
        if dataset == 'aro':
            for batch in data_loader:
                pos_txt, neg_txt, img = batch
                self.compile_batch(pos_txt)
                self.compile_batch(neg_txt)
                self.compile_batch(img)

    def fast_batch_contract(self, batch_recipes, training=False):
        groups = defaultdict(list)
        for i, (einsum_str, tensors) in enumerate(batch_recipes):
            groups[einsum_str].append((i, tensors))

        batch_size = len(batch_recipes)
        results = torch.zeros(batch_size, *[2]*self.qout, dtype=self.precision, device=self.device)
        for einsum_str, items in groups.items():
            indices, tensor_lists = zip(*items)
            minibatch_size = len(indices)
            gate_columns = zip(*tensor_lists)
            stacked_tensors = []
            shapes = []
            for column in gate_columns:
                first_symbol, op_type = column[0]
                if first_symbol is None:
                    if op_type == 'sqrt': shapes.append(torch.Size([minibatch_size]))
                    elif op_type == '0': shapes.append(torch.Size([minibatch_size, 2]))
                    elif op_type == 'H': shapes.append(torch.Size([minibatch_size, 2, 2]))
                    elif op_type == 'CX': shapes.append(torch.Size([minibatch_size, 2, 2, 2, 2]))
                else:
                    symbols = [symbol for symbol, _ in column]
                    weight_indices = [self.sym2weight[s] for s in symbols]
                    thetas = self.weights[weight_indices] 
                    if op_type == 'Rz': gate_batch = BatchRz(thetas, self.precision)
                    elif op_type == 'Rx': gate_batch = BatchRx(thetas, self.precision)
                    elif op_type == 'Ry': gate_batch = BatchRy(thetas, self.precision)
                    elif op_type == 'CRz': gate_batch = BatchCRz(thetas, self.precision)
                    elif op_type == 'CRx': gate_batch = BatchCRx(thetas, self.precision)
                    elif op_type == 'CRy': gate_batch = BatchCRy(thetas, self.precision)
                    shapes.append(gate_batch.shape)
                    stacked_tensors.append(gate_batch)

            input_subscripts, output_subscript = einsum_str.split('->')
            batched_inputs = ["$" + s for s in input_subscripts.split(',')]
            batched_str = f"{','.join(batched_inputs)}->${output_subscript}"
            cache_key = (batched_str, tuple(shapes))

            if cache_key in self.path_cache:
                group_out = self.path_cache[cache_key](*stacked_tensors)
            else:
                self.path_cache[cache_key] = contract_expression(batched_str, *shapes)
                group_out = self.path_cache[cache_key](*stacked_tensors)

            results[list(indices)] = group_out
        return results
    
    # add option to submsampe
    def get_embeddings_aro(self, data_loader):
        pos_txt_out = []
        neg_txt_out = []
        img_out = []
        for batch in data_loader:
            pos_circ, neg_circ, img = batch

            pos_txt = self.fast_batch_contract(pos_circ)
            neg_txt = self.fast_batch_contract(neg_circ)
            img = torch.stack([img_enc for img_enc in img])

            pos_txt = pos_txt.reshape(pos_txt.size(0), -1)
            pos_txt = F.normalize(pos_txt, dim=1, p=2)
            neg_txt = neg_txt.reshape(neg_txt.size(0), -1)
            neg_txt = F.normalize(neg_txt, dim=1, p=2)
            img = img.reshape(img.size(0), -1)

            pos_txt_out.append(pos_txt)
            neg_txt_out.append(neg_txt)
            img_out.append(img)
        pos_txt_out = torch.cat(pos_txt_out)
        neg_txt_out = torch.cat(neg_txt_out)
        img_out = torch.cat(img_out)
        return pos_txt_out, neg_txt_out, img_out

    def get_embeddings_svo(self, data_loader):
        txt_out = []
        pos_img_out = []
        neg_img_out = []
        for batch in data_loader:
            circ, pos_img_enc, neg_img_enc = batch

            txt = self.fast_batch_contract(circ)
            pos_img = torch.stack([img_enc for img_enc in pos_img_enc])
            neg_img = torch.stack([img_enc for img_enc in neg_img_enc])

            txt = txt.reshape(txt.size(0), -1)
            txt = F.normalize(txt, dim=1, p=2)
            pos_img = pos_img.reshape(pos_img.size(0), -1)
            neg_img = neg_img.reshape(neg_img.size(0), -1)

            txt_out.append(txt)
            pos_img_out.append(pos_img)
            neg_img_out.append(neg_img)
        txt_out = torch.cat(txt_out)
        pos_img_out = torch.cat(pos_img_out)
        neg_img_out = torch.cat(neg_img_out)
        return txt_out, pos_img_out, neg_img_out  
    
class QInfoNCE_dot(torch.nn.Module):
    def __init__(self, reduction='mean', device=None, eps=1e-7, lambda_reg=0.1):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.eps = eps
        self.lambda_reg = lambda_reg
    
    def forward(self, txt, img, T=0.07):
        B = txt.size(0)
        txt = F.normalize(txt.reshape(B, -1), dim=1, p=2)
        img = F.normalize(img.reshape(B, -1), dim=1, p=2)

        logits = (txt @ img.conj().t()).abs() / T
        labels = torch.arange(B, device=self.device)
        loss_txt = self.cross_entropy(logits, labels)
        loss_img = self.cross_entropy(logits.t(), labels)
        sym_loss =  (loss_txt + loss_img) / 2

        if self.lambda_reg > 0:
            txt_logits = (txt @ txt.conj().t()).abs()
            off_diag = txt_logits[~torch.eye(B, dtype=torch.bool, device=txt.device)]
            purity_penalty = off_diag.mean()
            return sym_loss + self.lambda_reg * purity_penalty
        return sym_loss

class QInfoNCE_cos(torch.nn.Module):
    def __init__(self, reduction='mean', device=None, eps=1e-7, lambda_reg=0.1):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.eps = eps
        self.lambda_reg = lambda_reg
    
    def forward(self, txt, img, T=0.07):
        B = txt.size(0)
        txt = F.normalize(txt.reshape(B, -1), dim=1, p=2)
        img = F.normalize(img.reshape(B, -1), dim=1, p=2)

        overlap = (txt @ img.conj().t()).abs()
        thetas = torch.acos(overlap.clamp(0, 1 - self.eps))
        logits = ((torch.full(thetas.size(), torch.pi/2) - thetas) / (torch.pi/2)) / T
        labels = torch.arange(B, device=self.device)
        loss_txt = self.cross_entropy(logits, labels)
        loss_img = self.cross_entropy(logits.t(), labels)
        sym_loss =  (loss_txt + loss_img) / 2.0

        if self.lambda_reg > 0:
            txt_overlap = (txt @ txt.conj().t()).abs()
            txt_thetas = torch.acos(txt_overlap.clamp(0, 1 - self.eps))
            txt_logits = (torch.full(txt_thetas.size(), torch.pi/2) - txt_thetas)
            off_diag = txt_logits[~torch.eye(B, dtype=torch.bool, device=txt.device)]
            purity_penalty = off_diag.mean()
            return sym_loss + self.lambda_reg * purity_penalty
        return sym_loss

def update_model_svo(model, data_loader, loss_fn, acc_fn, optimizer):
    cum_loss = cum_acc = cum_grad = 0
    model.train()
    for batch in data_loader:
        circ, pos_img, neg_img = batch

        pos_img = torch.stack([img_enc for img_enc in pos_img])
        neg_img = torch.stack([img_enc for img_enc in neg_img])
        pos_img = pos_img.reshape(pos_img.size(0), -1)
        neg_img = neg_img.reshape(neg_img.size(0), -1)

        optimizer.zero_grad()
        txt = model.fast_batch_contract(circ, True)
        txt = txt.reshape(txt.size(0), -1)
        loss = loss_fn(txt, pos_img, model.temp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        model.log_temp.data.clamp_(min=torch.log(torch.tensor(0.01)), max=torch.log(torch.tensor(1.0)))
        optimizer.step()
        cum_loss += loss.item()

        posres = acc_fn(pos_img, txt)
        negres = acc_fn(neg_img, txt)
        cum_acc += (torch.sum((posres > negres))/len(posres)).item()

        cum_grad += model.grad_norm

    N = len(data_loader)
    return (cum_loss/N, cum_acc/N, cum_grad/N)
    
def update_model_aro(model, data_loader, loss_fn, acc_fn, optimizer):
    cum_loss = cum_acc = cum_grad = 0
    model.train()
    for batch in data_loader:
        pos_circ, neg_circ, img = batch

        img = torch.stack([img_enc for img_enc in img])
        img = img.reshape(img.size(0), -1)
        neg_txt = model.fast_batch_contract(neg_circ, False)
        neg_txt = neg_txt.reshape(neg_txt.size(0), -1)

        optimizer.zero_grad()
        pos_txt = model.fast_batch_contract(pos_circ, True)
        pos_txt = pos_txt.reshape(pos_txt.size(0), -1)
        loss = loss_fn(pos_txt, img, model.temp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.log_temp.data.clamp_(min=torch.log(torch.tensor(0.01)), max=torch.log(torch.tensor(1.0)))
        optimizer.step()
        cum_loss += loss.item()

        posres = acc_fn(pos_txt, img)
        negres = acc_fn(neg_txt, img)
        cum_acc += (torch.sum((posres > negres))/len(posres)).item()

        cum_grad += model.grad_norm

    N = len(data_loader)
    return (cum_loss/N, cum_acc/N, cum_grad/N)

def update_model_aro2(model, data_loader, loss_fn, acc_fn, optimizer):
    cum_loss = cum_acc = cum_grad = 0
    model.train()
    for batch in data_loader:
        pos_circ, neg_circ, img_circ = batch

        img_vec = model.fast_batch_contract(img_circ, False)
        img_vec = img_vec.reshape(img_vec.size(0), -1)
        neg_txt = model.fast_batch_contract(neg_circ, False)
        neg_txt = neg_txt.reshape(neg_txt.size(0), -1)

        optimizer.zero_grad()
        pos_txt = model.fast_batch_contract(pos_circ, True)
        pos_txt = pos_txt.reshape(pos_txt.size(0), -1)
        loss = loss_fn(pos_txt, img_vec, model.temp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model.log_temp.data.clamp_(min=torch.log(torch.tensor(0.01)), max=torch.log(torch.tensor(1.0)))
        optimizer.step()
        cum_loss += loss.item()

        posres = acc_fn(pos_txt, img_vec)
        negres = acc_fn(neg_txt, img_vec)
        cum_acc += (torch.sum((posres > negres))/len(posres)).item()

        cum_grad += model.grad_norm

    N = len(data_loader)
    return (cum_loss/N, cum_acc/N, cum_grad/N)

def eval_model_svo(model, data_loader, acc_fn):
    cum_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            circ, pos_img, neg_img = batch

            pos_img = torch.stack([img_enc for img_enc in pos_img])
            neg_img = torch.stack([img_enc for img_enc in neg_img])
            txt = model.fast_batch_contract(circ, False)

            pos_img = pos_img.reshape(pos_img.size(0), -1)
            neg_img = neg_img.reshape(neg_img.size(0), -1)
            txt = txt.reshape(txt.size(0), -1)

            posres = acc_fn(pos_img, txt)
            negres = acc_fn(neg_img, txt)
            cum_acc += (torch.sum((posres > negres))/len(posres)).item()
    
    return cum_acc/len(data_loader)

def eval_model_aro(model, data_loader, acc_fn):
    cum_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            pos_circ, neg_circ, img = batch

            pos_txt = model.fast_batch_contract(pos_circ, False)
            neg_txt = model.fast_batch_contract(neg_circ, False)
            img = torch.stack([img_enc for img_enc in img])

            pos_txt = pos_txt.reshape(pos_txt.size(0), -1)
            neg_txt = neg_txt.reshape(neg_txt.size(0), -1)
            img = img.reshape(img.size(0), -1)

            posres = acc_fn(img, pos_txt)
            negres = acc_fn(img, neg_txt)
            cum_acc += (torch.sum((posres > negres))/len(posres)).item()
    
    return cum_acc/len(data_loader)

def eval_model_aro2(model, data_loader, acc_fn):
    cum_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            pos_circ, neg_circ, img_circ = batch

            pos_txt = model.fast_batch_contract(pos_circ, False)
            neg_txt = model.fast_batch_contract(neg_circ, False)
            img_vec = model.fast_batch_contract(img_circ, False)

            pos_txt = pos_txt.reshape(pos_txt.size(0), -1)
            neg_txt = neg_txt.reshape(neg_txt.size(0), -1)
            img_vec = img_vec.reshape(img_vec.size(0), -1)

            posres = acc_fn(img_vec, pos_txt)
            negres = acc_fn(img_vec, neg_txt)
            cum_acc += (torch.sum((posres > negres))/len(posres)).item()
    
    return cum_acc/len(data_loader)