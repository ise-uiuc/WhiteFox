
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 4
        self.seq_len = 64
        self.dim = 264
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.3, True)
        output = attn_weight @ value
        output = output.transpose(0, 2).view(1, 64, 16, 256)
        return output
# Inputs to the model
query = torch.randn(1, 4, 64, 264)
key = torch.randn(1, 4, 64, 264)
value = torch.randn(1, 4, 64, 264)
attn_mask = torch.randn(1, 1, 64, 64)
