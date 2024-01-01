
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        x = self.conv1(x1)
        x = self.conv2(x)
        x = self.conv3(x)
        v1 = torch.neg(x)
        v1 = torch.clamp(v1, 0, 1)
        v1 = torch.mean(v1, dim=0, keepdim=True)
        return v1
# Inputs to the model
x1 = torch.randn(3, 3, 224, 224)
