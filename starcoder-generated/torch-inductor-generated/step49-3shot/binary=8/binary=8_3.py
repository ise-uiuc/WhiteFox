
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)        
        self.conv4 = torch.nn.Conv2d(8, 4, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = x1 + v1
        v4 = x2 + v2
        v4 = self.conv3(v4)
        v5 = self.conv4(v3)
        return v5
# Input to the model
x1 = torch.randn(1, 16, 16, 16)
x2 = torch.randn(1, 16, 16, 16)
