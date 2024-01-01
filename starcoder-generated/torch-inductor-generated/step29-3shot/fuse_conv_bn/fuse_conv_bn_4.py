
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(5, 5, 2), torch.nn.BatchNorm2d(5))
    def forward(self, x):
        out = self.layer(x)
        return (out, out, out)
# Inputs to the model
x = torch.randn(1, 5, 4, 4)
