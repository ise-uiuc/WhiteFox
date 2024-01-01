
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    def forward(self, x):
        y1 = self.linear(x)
        return torch.cat((y1.relu(), y1.relu()), dim=0)
# Inputs to the model
x = torch.randn(2, 2)
# model ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x):
        return torch.cat((x, x), dim=1).view(x.shape[0], -1)
# Inputs to the model
x = torch.randn(2, 2)
# model ends