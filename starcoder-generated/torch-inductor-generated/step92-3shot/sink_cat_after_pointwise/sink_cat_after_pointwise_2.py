
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = torch.randn(2, 3, 4)
        a2 = torch.randn(1, 3, 4)
        x = torch.cat([a1, a2], dim=0)
        y = torch.relu(x)
        return y.view(y.shape[0], *y.shape[2:])
# Inputs to the model
x = torch.randn(3, 3, 4)
