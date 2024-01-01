
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        y = torch.cat([x, x], dim=1)
        return torch.relu(y)
# Inputs to the model
x = torch.randn(2, 3, 4)
y = torch.randn(2, 5)
