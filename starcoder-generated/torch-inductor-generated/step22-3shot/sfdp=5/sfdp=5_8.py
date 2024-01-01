
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.1
        self.head = 8
        self.seq_len = 32768
        self.dim = 128 // self.head
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 768, 32768, 128)
key = torch.randn(1, 768, 32768, 128)
value = torch.randn(1, 768, 32768, 64)
attn_mask = torch.randn(1, 1, 32768, 32768)
