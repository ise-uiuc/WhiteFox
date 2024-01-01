
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(3, 50, 3, stride=2, padding=1), torch.nn.BatchNorm2d(50))
    def forward(self, x1):
        v1 = self.layer(x1)
        return torch.sigmoid(v1)
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
