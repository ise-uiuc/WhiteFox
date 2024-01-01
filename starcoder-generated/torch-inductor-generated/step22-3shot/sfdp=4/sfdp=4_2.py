
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = x1 @ x2.transpose(-2, -1)
        v1 = v1 / math.sqrt(x1.size(-1))
        v1 = v1 + attn_mask
        v2 = attn_weight @ x1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 400)
attn_mask = torch.randn(1, 1, 400)
attn_weight = torch.randn(1, 1, 400)
