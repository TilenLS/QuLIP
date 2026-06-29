import torch, mlflow, time, yaml, argparse, os
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from pathlib import Path

from factory import build_experiment
from modules.utils.general import load_pkl, store_pkl, get_device, set_seed
from modules.data_pipeline.datasets import CocoDataset, coco_collate_fn
from modules.models.fusion.engine import ContrastiveTrainer, MMEvaluator, coco_mapper

ROOT_PATH = os.getcwd()
# uv run python train.py --config configs/tensor_network.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config YAML')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    DEV = get_device()
    BATCH_SIZE = config['batch_size']
    DATASET = config['dataset']

    ansatz, image_model, text_model, loss_fn = build_experiment(config, DEV)
    
    df_train = load_pkl(config['splits']['train']['text_path'])
    df_val = load_pkl(config['splits']['val']['text_path'])
    compiled_train = ansatz.compile_dataset(df_train)
    compiled_val = ansatz.compile_dataset(df_val)

    if hasattr(text_model, "from_symbols"):
        cols = [col for col in compiled_train.columns if col.endswith('_symbols')]
        txt_stream = []
        for col in cols:
            txt_stream += compiled_train[col].tolist() + compiled_val[col].tolist()
        text_model.from_symbols(txt_stream)

    
    if hasattr(image_model, "fit_image_pca"):
        train_embeddings = torch.load(config['splits']['train']['img_path'])
        image_model.fit_image_pca(torch.stack(list(train_embeddings.values())).to(DEV))
        img_transform = None

    if config['vision']['use_clip']:
        img_transform = None
    else:
        img_transform = v2.Compose([v2.ToImage(),               
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize((64, 64))])

    train_dataset = CocoDataset(compiled_train, config['splits']['train']['img_path'], image_transform=img_transform, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=coco_collate_fn, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = CocoDataset(compiled_val, config['splits']['val']['img_path'], image_transform=img_transform, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=coco_collate_fn, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(list(text_model.parameters()) + list(image_model.parameters()), lr=config['learning_rate'])
    trainer = ContrastiveTrainer(image_model, text_model, optimizer, loss_fn, DEV)
    evaluator = MMEvaluator(image_model, text_model, DEV)

    SEED = int.from_bytes(os.urandom(4))
    set_seed(SEED)

    run_name = f"{int(time.time())}"
    checkpoint_path = f"./checkpoints/{DATASET}/{run_name}.pt"
    Path(os.path.dirname(checkpoint_path)).mkdir(parents=True, exist_ok=True)
    mlf_path = os.path.join(ROOT_PATH, f'mlf_dbs/{DATASET}.db')

    mlflow.pytorch.autolog()
    mlflow.set_tracking_uri(f"sqlite:///{mlf_path}")
    mlflow.set_experiment(DATASET)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": config['epochs'],
            "batch_size": config['batch_size'],
            "learning_rate": config['learning_rate'],
            "temperature_parameter": loss_fn.temperature,
            "device_target": str(DEV),
            "seed": SEED,
            "image_tower": type(image_model).__name__,
            "text_tower": type(text_model).__name__,
            })
        
        for epoch in range(config['epochs']):
            start_time = time.time()
            loss = trainer.train_epoch(train_loader, coco_mapper)
            end_time = time.time()
            
            metrics = evaluator.global_retrieval(val_loader, coco_mapper)
            
            mlflow.log_metrics(metrics, step=epoch)
            output_str = [f"{k}: {v:.4f}" for k, v in metrics.items()]
            print(f"Epoch {epoch} | Loss: {loss:.4f} | Metrics: {', '.join(output_str)} | Time: {end_time - start_time:.2f}s")

            torch.save({
                "image": image_model.state_dict(),
                "text": text_model.state_dict(),
                "epoch": epoch,
                "train_loss": loss,
                "val_metrics": metrics if val_loader else None
                }, checkpoint_path)