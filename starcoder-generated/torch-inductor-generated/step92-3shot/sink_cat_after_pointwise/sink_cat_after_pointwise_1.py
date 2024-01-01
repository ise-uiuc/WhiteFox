
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, torch.cat([x, x], dim=1)], dim=0).view(x.shape[0], -1)
        return torch.cat([x, x], dim=0)
# Inputs to the model
x = torch.randn(2, 3, 4)
