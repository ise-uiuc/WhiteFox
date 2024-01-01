
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.ReLU6())
    def forward(self, x2):
        return self.layer(x2) + 5
# Inputs to the model
x2 = torch.randn(1, 4, 4, 4)
