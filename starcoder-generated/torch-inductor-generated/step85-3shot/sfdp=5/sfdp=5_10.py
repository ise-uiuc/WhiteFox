
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 196
        self.seq_len = 455
        self.dim = 647 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.36, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 196, 455, 647)
key = torch.randn(1, 196, 455, 647)
value = torch.randn(1, 196, 455, 647)
attn_mask = torch.randn(1, 1, 455, 455)
