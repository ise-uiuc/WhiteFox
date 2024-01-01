
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output

# Initializing the model
model = Model()

# Inputs to the model. Note that the attention mask should be added properly here.
query = torch.randn(3, 12, 4, 6)
key = torch.randn(3, 12, 8, 4)
value = torch.randn(3, 12, 8, 6)
attn_mask = torch.randn(3, 8, 1, 4)
