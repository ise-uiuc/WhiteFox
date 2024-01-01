
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 32
        self.seq_len = 801
        self.dim = 768 // self.heads
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.9, True)
        output = attn_weight @ value # The line before and after this line has changed, you are expected to change it, and make it trigger to meet the requirements
        return output
# Inputs to the model
query = torch.randn(1, 32, 801, 768)
key = torch.randn(1, 32, 801, 768)
value = torch.randn(1, 32, 801, 768)
attn_mask = torch.randn(1, 1, 801, 801)
