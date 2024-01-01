
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 8
        self.seq_len = 1024
        self.seq_len_2 = 128
        self.dim = 784 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.1, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 8, 768, 784)
key = torch.randn(1, 8, 768, 784)
value = torch.randn(1, 8, 768, 784)
attn_mask = torch.randn(1, 1, 768, 768)
