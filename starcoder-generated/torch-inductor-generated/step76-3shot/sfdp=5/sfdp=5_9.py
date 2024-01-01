
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 1
        self.seq_len = 3
        self.dim = 2
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=0)
        attn_weight = torch.dropout(attn_weight, 0.6, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(3, 1, 2)
key = torch.randn(3, 1, 2)
value = torch.randn(3, 1, 2)
attn_mask = torch.tensor([[0, 0, 1], [1, 1, 0], [1, 1, 1]])
