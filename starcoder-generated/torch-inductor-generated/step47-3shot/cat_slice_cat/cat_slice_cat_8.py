
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.cat((y, x, x), 1)
        v2 = v1[:, -1]
        v3 = v1[:, :2]
        v4 = torch.cat((v3, v2))
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)
y = torch.randn(1, 3)
