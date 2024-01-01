
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        v1 = x1 @ x2.transpose(-2, -1)
        v2 = v1 / math.sqrt(x1.size(-1))
        v3 = v2 + x3
        v4 = torch.softmax(v3, dim=-1)
        output = v4 @ x2
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(96, 16, 256)
x2 = torch.randn(96, 16, 256) 
x3 = torch.arange(96)
