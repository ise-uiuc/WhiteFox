
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(64, 128, 7, stride=2, padding=3)
        self.conv3 = torch.nn.Conv2d(128, 256, 7, stride=2, padding=3)
        self.conv4 = torch.nn.Conv2d(256, 512, 7, stride=2, padding=3)
    
    def forward(self, x1):
        v1 = F.relu_(self.conv1(x1))
        v2 = F.relu_(self.conv2(v1))
        out = F.relu_(self.conv3(v2))
        out1 = F.relu_(self.conv4(out))
        return out1

# Test inference
m = Model()
m.eval()
input = torch.randn(3, 3, 64, 64)
