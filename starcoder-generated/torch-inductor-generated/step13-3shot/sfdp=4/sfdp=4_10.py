
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = x1 @ x2.transpose(-1, -2)
        v2 = v1 / math.sqrt(x1.size(-1))
        v3 = v2 + x3
        v4 = F.softmax(v3, dim=-1)
        v5 = v4 @ x3
        return v5

# Initializing the model
m = Model()

# Inputs to the model
__input1__ = torch.randn(1, 2, 3)
__input2__ = torch.randn(1, 3, 2)
__input3__ = torch.randn(1, 3, 4)
