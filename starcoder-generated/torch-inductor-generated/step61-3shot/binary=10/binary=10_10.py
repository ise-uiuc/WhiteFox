 (same as https://github.com/pytorch/benchmark/blob/239c969c5288fc28b846894440ef1d154c40ba04/models/BERT/bert_pytorch.py)

class BERTLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        