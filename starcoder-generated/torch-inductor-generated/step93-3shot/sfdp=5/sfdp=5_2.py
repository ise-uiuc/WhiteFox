
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 32
        self.seq_len = 2048
        self.dim = 512 // self.heads
    def forward(self, query):
        key = torch.randn(1, self.heads, 2048, self.dim)
        value = torch.randn(1, self.heads, 2048, self.dim)
        attn_mask = torch.randn(1, 1, 2048, 2048)
        return Model()(query, key, value, attn_mask), key, value
# Inputs to the model
query = torch.randn(1, 16, 128, 256)
