
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.maxpool = torch.nn.AvgPool2d(3, 3, 3) 
    def forward(self, v1):
        out = self.features(v1)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = out.view(1, -1)
        return (out, torch.split(v1, 64, 1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
