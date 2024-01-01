
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(20, 5, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(15, 10, 1, stride=1, padding=1)
    def forward(self, x1, some_parameter=None):
        v1 = self.conv1(x1[:5].clone())
        v2 = self.conv2(x1+v1)
        v3 = self.conv3(v1+v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 10, 64, 64)
