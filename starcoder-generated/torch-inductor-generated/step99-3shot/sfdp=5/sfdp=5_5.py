
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 35
        self.seq_len = 2048
        self.dim = 64
    def forward(self, query, key, pos_embedding, attn_mask):
        