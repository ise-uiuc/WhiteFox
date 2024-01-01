
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv4 = torch.nn.Conv2d(3, 3, 3, bias=False)
        self.conv4.bias = torch.nn.Parameter(torch.randn(3))
        self.batch_norm4 = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        x = self.conv4(x)
        x = self.batch_norm4(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
