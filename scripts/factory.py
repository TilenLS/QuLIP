# factory.py
from modules.compilation.quantum.ansatz import IQPAnsatz, CustomV5Ansatz
from modules.compilation.classical.tensor_network import TNCompiler
from modules.compilation.classical.decompositions import MPS
from modules.compilation.classical.neural import MLPCompiler

from modules.models.text.einsum_quantum import QCModel
from modules.models.text.einsum_classical import TNModel
from modules.models.text.neural_model import MLPModel

from modules.models.vision.quantum_map import QuantumFeatureMap, FrozenCLIP
from modules.models.vision.image_model import TTNImageModel

from modules.models.fusion.criteria import FS_InfoNCE, InfoNCE

def build_experiment(config, device):
    model_type = config['model_type']
    
    if model_type == 'tn':
        # 1. Compiler
        compiler = config['compiler']
        obmap = {'n': compiler['n'], 's': compiler['s'], 'p': compiler['p'], 'out': config['embedding_dim']}
        mps_proc = MPS(bond_dim=config['compiler']['bond_dim'], max_order=config['compiler']['max_order'])
        ansatz = TNCompiler(obmap=obmap, decomp_fn=mps_proc)
        
        # 2. Models
        if config['vision']['use_clip']:
            image_model = FrozenCLIP().to(device)
        else:
            image_model = TTNImageModel(embedding_dim=config['embedding_dim']).to(device)
        text_model = TNModel(out_dim=config['embedding_dim']).to(device)
        
        # 3. Loss
        loss_fn = InfoNCE()
        
    elif model_type == "vqc":
        # 1. Compiler
        compiler = config['compiler']
        obmap = {'n': compiler['n'], 's': compiler['s'], 'p': compiler['p'], 'out': config['embedding_qubits']}
        ansatz = CustomV5Ansatz(obmap=obmap, layers=compiler['layers'])
        
        # 2. Models
        if config['vision']['use_clip']:
            image_model = FrozenCLIP().to(device)
        else:
            image_model = QuantumFeatureMap(k=compiler['embedding_qubits'], layers=config['vision']['layers']).to(device)
        text_model = QCModel(out_q=config['compiler']['out']).to(device)
        
        # 3. Loss
        loss_fn = FS_InfoNCE()
        
    elif model_type == "mlp":
        ansatz = MLPCompiler()
        if config['vision']['use_clip']:
            image_model = FrozenCLIP().to(device)
        else:
            image_model = TTNImageModel(embedding_dim=config['embedding_dim']).to(device)
        text_model = MLPModel(out_dim=config['embedding_dim']).to(device)
        loss_fn = InfoNCE()
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return ansatz, image_model, text_model, loss_fn