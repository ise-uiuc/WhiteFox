
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_mask_dim = 12
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask # Add a dimension where is to be filled with ones
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.2, True)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 32, 256, 128)
key = torch.randn(1, 32, 256, 128)
value = torch.randn(1, 32, 256, 128)
attn_mask = torch.ones(1, 1, 256, 12)
