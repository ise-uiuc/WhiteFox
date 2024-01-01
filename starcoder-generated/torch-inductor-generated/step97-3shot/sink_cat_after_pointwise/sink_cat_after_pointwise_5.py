
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=3)
        y = y if y.shape[2:] == torch.Size((1, 3)) else torch.relu(y)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
