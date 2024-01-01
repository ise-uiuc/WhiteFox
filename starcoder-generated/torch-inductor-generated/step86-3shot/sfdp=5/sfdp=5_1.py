
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 64
        self.seq_len = 2048
        self.dim = 256 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight1 = torch.softmax(qk, dim=-1)
        attn_weight1 = torch.dropout(attn_weight1, 0.9, True)
        attn_weight2 = torch.softmax(qk, dim=-2)
        attn_weight2 = torch.dropout(attn_weight2, 0.9, True)
        attn_weight3 = torch.softmax(qk, dim=0)
        attn_weight3 = torch.dropout(attn_weight3, 0.9, True)
        output = attn_weight3 @ value @ attn_weight1 @ attn_weight2
        return output
# Inputs to the model
query = torch.randn(1, 64, 2048, 256)
key = torch.randn(1, 64, 2048, 256)
value = torch.randn(1, 64, 2048, 256)
attn_mask = torch.randn(1, 1, 2048, 2048)
