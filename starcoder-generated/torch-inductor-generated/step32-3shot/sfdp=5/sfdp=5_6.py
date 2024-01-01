
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 32
        self.seq_len = 128
        self.dim = 768
    def forward(self, query, key, value):
        qk = query @ key.transpose(-2, -1)
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 32, 128, 768)
key = torch.randn(1, 32, 128, 768)
value = torch.randn(1, 32, 128, 768)
