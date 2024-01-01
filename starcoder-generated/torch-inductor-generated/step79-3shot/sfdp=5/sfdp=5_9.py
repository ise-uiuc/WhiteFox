
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 7
        self.seq_len = 16
        self.dim = 96 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = attn_mask @ query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 7, 16, 96)
key = torch.randn(1, 7, 16, 96)
value = torch.randn(1, 7, 16, 96)
attn_mask = torch.randn(1, 1, 16, 16)
