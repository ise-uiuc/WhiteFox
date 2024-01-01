
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, (1, 2), stride=(1, 1), padding=(0, 1), groups=1, bias=False)
        self.conv2 = torch.nn.Conv2d(8, 1, (2, 1), stride=(1, 1), padding=(1, 0), groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.pad(v1, (0, 0, 0, 0, 0, 1), "constant", 0.4)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 64)
