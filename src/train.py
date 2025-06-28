
import torch
from torch.utils.data import DataLoader
from src.model import SimpleTTSModel, save_model
from src.data import TTSDataset
import os
from omegaconf import OmegaConf

def train_loop(config_path, data_path, output_dir):
    config = OmegaConf.load(config_path)
    dataset = TTSDataset(data_path, sample_rate=config.data.sample_rate)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)
    model_args = config.model.params
    model = SimpleTTSModel(**model_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    criterion = torch.nn.MSELoss()
    os.makedirs(output_dir, exist_ok=True)
    for epoch in range(config.training.epochs):
        model.train()
        for audio, text in loader:
            # TODO: Replace with real text-to-seq and mel processing
            x = torch.randint(0, 255, (audio.shape[0], 50))
            y = torch.randn(audio.shape[0], 50, model_args['hidden_size'])
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % config.training.save_every == 0:
            save_model(model, os.path.join(output_dir, f"model_{epoch+1}.pth"), model_args)
        print(f"Epoch {epoch+1}/{config.training.epochs} done.")

def add_args(parser):
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output', type=str, default='models/', help='Output directory for checkpoints')

def run(args):
    train_loop(args.config, args.data, args.output)
