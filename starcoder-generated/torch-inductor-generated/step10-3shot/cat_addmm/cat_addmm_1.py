
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3):
        x1 = torch.mm(x2, x3)
        x2 = torch.add(x1, x2)
        x3 = torch.cat([x2], 0)
        return x3, x1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(10, 7)
x3 = torch.randn(7, 10)
