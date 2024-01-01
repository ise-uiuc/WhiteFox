
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.seq_len = 512
        self.dim = 128
        self.depth = 4
    