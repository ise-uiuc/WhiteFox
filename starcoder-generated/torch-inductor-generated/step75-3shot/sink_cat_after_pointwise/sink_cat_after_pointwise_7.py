
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(2, 3)
        x = torch.cat([y, x], dim=1)
        return x.view(3, -1)
# Inputs to the model
x = torch.randn(3, 2)
