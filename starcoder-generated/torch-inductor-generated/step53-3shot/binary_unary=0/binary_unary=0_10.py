
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.identity1 = torch.nn.Identity()
        self.identity2 = torch.nn.Identity()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        t1 = self.identity1(x)
        t2 = self.conv1(t1)
        v1 = self.identity2(t2)
        v2 = self.conv2(v1)
        v3 = v2 + v1
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
