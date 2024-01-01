
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 4096
        self.seq_len = 2
        self.dim = 128
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.6050939852564436, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 4608, 256, 128)
key = torch.randn(1, 4608, 256, 128)
value = torch.randn(1, 4608, 256, 128)
attn_mask = torch.randn(1, 1, 256, 256)
