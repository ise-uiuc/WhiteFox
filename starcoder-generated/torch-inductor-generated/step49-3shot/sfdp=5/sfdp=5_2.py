
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 65
        self.seq_len = 7500
        self.dim = 512 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=0)
        attn_weight = torch.dropout(attn_weight, 0.2, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(65, 1, 7500, 512)
key = torch.randn(65, 1, 7500, 512)
value = torch.randn(65, 1, 7500, 512)
attn_mask = torch.randn(1, 65, 7500, 7500)
