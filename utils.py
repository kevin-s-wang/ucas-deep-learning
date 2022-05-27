
import torch

def get_device():
    # TODO: support Apple GPUs
    return 'cuda' if torch.cuda.is_available() else 'cpu'