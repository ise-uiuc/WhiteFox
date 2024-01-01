
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torch.nn.MaxPool2d(2)
        c = torch.nn.Conv2d(3, 3, 3)
        self.layer = torch.nn.Sequential(m, c)
    def forward(self, x):
        c = self.layer(x)
        return c, x
# Inputs to the model
x = torch.randn(2, 3, 4, 4)
