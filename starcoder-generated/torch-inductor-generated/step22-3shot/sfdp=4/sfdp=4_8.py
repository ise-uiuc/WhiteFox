
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, attn_mask):
        s1 = query @ key.transpose(-2, -1)
        s2 = s1 / math.sqrt(query.size(-1))
        s2 = s2 + attn_mask
        output = torch.nn.functional.softmax(s2, dim=-1) @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
__query__ = torch.randn(1, 6, 7)
__key__ = torch.randn(1, 5, 7)
__value__ = torch.randn(1, 5, 6)
__attn_mask__ = torch.randn(1, 5, 6)

# Output of the model
output = m(__query__, __key__, __value__, __attn_mask__)
