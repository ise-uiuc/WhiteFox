
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 67
        self.seq_len = 173
        self.dim = 78 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.032, True)
        output = attn_weight @ value
        return output
# Input to the model
query = torch.randn(1, 67, 173, 78)
key = torch.randn(1, 67, 173, 78)
value = torch.randn(1, 67, 173, 78)
attn_mask = torch.randn(1, 1, 173, 173)
