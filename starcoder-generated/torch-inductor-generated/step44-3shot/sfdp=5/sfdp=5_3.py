
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 64
        self.seq_len = 178
        self.dim = 320 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.<KEY>())
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 16, 178, 320)
key = torch.randn(1, 64, 178, 320)
value = torch.randn(1, 64, 178, 320)
attn_mask = torch.randn(1, 2, 178, 178)
