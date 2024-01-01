
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = F.relu(x)
        y = torch.cat([x, x, x], dim=1)
        y = torch.tanh(y)
        x = torch.cat([y, y, y], dim=1)
        return x - y
# Inputs to the model
x = torch.randn(2, 3, 4)
