
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v = torch.cat([x, x], dim=1)
        return v.view(1, -1)
# Inputs to the model
x = torch.randn(1, 2, 2)
