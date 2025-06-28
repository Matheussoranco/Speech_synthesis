import torch
import torch.nn as nn

class SimpleTTSModel(nn.Module):
    """
    A minimal TTS model placeholder (replace with YourTTS/Chatterbox logic).
    """
    def __init__(self, input_dim=256, hidden_size=256, output_dim=80, num_layers=4, speaker_embedding=True):
        super().__init__()
        self.speaker_embedding = speaker_embedding
        self.embedding = nn.Embedding(256, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x, speaker_emb=None):
        x = self.embedding(x)
        if self.speaker_embedding and speaker_emb is not None:
            x = x + speaker_emb.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out

# Model loader

def load_model(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleTTSModel(**checkpoint['model_args'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

def save_model(model, path, model_args):
    torch.save({'state_dict': model.state_dict(), 'model_args': model_args}, path)
