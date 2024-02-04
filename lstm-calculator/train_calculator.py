import torch


def get_device():
    # Check for MPS (Apple Silicon GPU) availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device




def main():
    device = get_device()
    print("using device ", device)


main()
