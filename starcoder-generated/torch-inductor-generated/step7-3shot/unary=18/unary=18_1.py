
class Model(torch.nn.Module):
  
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 3, 1, stride=1, padding=1)
  
    def forward(self, x1):
        v0 = x1
        v1 = self.conv1(v0)
        v2 = torch.nn.Sigmoid()(v1)
        v3 = self.conv2(v2)
        v4 = torch.nn.Sigmoid()(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
