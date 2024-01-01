
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(2, 2)
        self.maxpool2 = torch.nn.MaxPool2d(2, 2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = v1.add(v2)
        v5 = v3.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 32, 32)
