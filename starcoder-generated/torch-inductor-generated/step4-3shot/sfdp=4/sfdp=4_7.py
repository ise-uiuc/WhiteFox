
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, attn_mask):
        v1 = x1 @ x2.transpose(-2, -1)
        v2 = v1 / math.sqrt(x1.size(-1))
        v3 = v2 + attn_mask
        v4 = torch.softmax(v3, dim=-1)
        v5 = v4 @ x3
        return v5
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 512)
x2 = torch.randn(2, 23, 32)
attn_mask = torch.randn(2, 1, 23)
