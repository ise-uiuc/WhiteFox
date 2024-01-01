
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x,x + 1.0), dim=1)
        y = torch.relu(y)
        return y
# Inputs to the model
x = torch.randn(2, 2, 2)
