
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = 0.1
        self.heads = 32
        self.seq_len = 128
        self.dim = 128 // self.heads
        self.linear0 = torch.nn.Linear(128, 128, bias=False)
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.dropout, True)
        output = attn_weight @ value
        output += query + key + value
        output = self.linear0(output)
        return output
# Inputs to the model
query = torch.randn(1, 32, 128, 128)
key = torch.randn(1, 32, 128, 128)
value = torch.randn(1, 32, 128, 128)
attn_mask = torch.randn(1, 1, 128, 128)
