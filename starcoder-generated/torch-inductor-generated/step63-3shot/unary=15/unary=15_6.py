
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(2, stride=2)
        self.pad = torch.nn.ZeroPad2d(1)
    def forward(self, x1):
        v1 = self.maxpool(x1)
        v2 = self.pad(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
