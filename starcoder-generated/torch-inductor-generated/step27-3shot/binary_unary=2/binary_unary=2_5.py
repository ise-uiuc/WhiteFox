
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = torch.cat((x1, x2),1)
        v2 = self.conv1(v1)
        v3 = v2 - 203
        v4 = F.relu(v3)
        t = torch.zeros_like(x1)
        return t
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 23, 16, 16)
