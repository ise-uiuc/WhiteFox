
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, attn_mask):
        attn_weight = torch.softmax((query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1)
        return attn_weight @ value

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 3)
key = torch.randn(1, 2, 4)
value = torch.randn(1, 2, 4)
attn_mask = torch.ones(1, 2, 3, 4)
