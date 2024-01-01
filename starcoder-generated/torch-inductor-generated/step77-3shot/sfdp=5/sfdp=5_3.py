
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 45
        self.seq_len = 294
        self.dim = 64 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.3, True)
        output = attn_weight @ value
        output = output.transpose(0, 2)
        return output
# Inputs to the model
query = torch.randn(256, 45, 294, 64)
key = torch.randn(256, 45, 294, 64)
value = torch.randn(256, 45, 294, 64)
attn_mask = torch.randn(1, 1, 294, 294)
