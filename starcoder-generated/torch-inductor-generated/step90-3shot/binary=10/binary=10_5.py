
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, b=torch.zeros(2)):
        v1 = torch.nn.functional.linear(x, torch.arange(9.).view(3, 3), bias=b)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(6, 3)
