
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2)
    def forward(self, x):
        y = x.view(x.size(0), -1)
        z = x.sum(dim=1) + 5
        x = torch.cat((y, z), dim=1)
        y = x.view(9, 9)
        x = x + torch.randn(2, 2)
        x = self.conv(x)
        return y
# Inputs to the model
x = torch.randn(1, 3, 2, 2)
