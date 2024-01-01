
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 64
        self.dim = 64
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 64, 64, 64)
key = torch.randn(1, 64, 64, 64)
value = torch.randn(1, 64, 64, 64)
attn_mask = torch.randn(1, 1, 64, 64)
