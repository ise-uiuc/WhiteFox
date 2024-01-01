
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.1
        self.heads = 4
        self.seq_len = 64
        self.dim = 16 // self.heads
        self.head_dim = self.dim * self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1)
        qk = qk / math.sqrt(self.dim)
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        output = output.view(-1, self.seq_len, self.head_dim)
        return output
# Inputs to the model
query = torch.randn(1, 8, 64, 16)
key = torch.randn(1, 8, 64, 16)
value = torch.randn(1, 8, 64, 16)
attn_mask = torch.randn(1, 1, 64, 64)
