
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        y = torch.relu(y)
        y = y.permute(0, 2, 1)
        y = torch.relu(y)
        return y
# Inputs to the model
x = torch.randn(1, 2, 3, 4)
