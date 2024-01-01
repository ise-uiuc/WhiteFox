
class Model(torch.nn.Module):
    def __init__(self):
        self.seq_len = 128
        self.heads = 32
        self.dim = 20 // self.heads
        super().__init__()
    def forward(query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 32, 128, 20)
key = torch.randn(1, 32, 128, 20)
value = torch.randn(1, 32, 128, 20)
attn_mask = torch.randn(1, 1, 128, 128)
