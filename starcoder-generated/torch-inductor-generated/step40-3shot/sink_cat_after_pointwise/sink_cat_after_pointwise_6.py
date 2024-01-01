
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        y = y.view(y.size(0), -1)
        return torch.relu(y)
# Inputs to the model
x = torch.randn(2, 3, 4)
