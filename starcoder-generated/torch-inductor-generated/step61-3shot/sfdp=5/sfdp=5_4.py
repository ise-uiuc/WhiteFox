
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 793
        self.seq_len = 741
        self.dim = 1060 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.6, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 256, 741, 1060)
key = torch.randn(1, 256, 741, 1060)
value = torch.randn(1, 256, 741, 1060)
attn_mask = torch.randn(1, 1, 741, 741)
