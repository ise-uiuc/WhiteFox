
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1024
        self.seq_len = 128
        self.dim = 64 // self.heads
    def forward(self, query, key, value):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 1024, 128, 64)
key = torch.randn(1, 1024, 128, 64)
value = torch.randn(1, 1024, 128, 64)
