
 class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 4
        self.seq_len = 32
        self.dim = 2048 // self.heads * self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 4, 32, 2048)
key = torch.randn(1, 4, 32, 2048)
value = torch.randn(1, 4, 32, 2048)
attn_mask = torch.randn(1, 1, 32, 32)
