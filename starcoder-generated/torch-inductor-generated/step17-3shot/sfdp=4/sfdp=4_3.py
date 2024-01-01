
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = x1 @ x2.transpose(-2, -1)
        v2 = v1 / math.sqrt(x1.size(-1))
        v3 = v2 + 0
        v4 = torch.softmax(v3, -1)
        v5 = v4 @ x2
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3, 32, 64)
x2 = torch.randn(2, 4, 32, 64)
