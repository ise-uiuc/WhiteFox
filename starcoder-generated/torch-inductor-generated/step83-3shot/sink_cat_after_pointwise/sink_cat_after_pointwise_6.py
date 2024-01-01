
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.size(0), -1)
        x = torch.stack((y, y, y), dim=2).sum(dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
