
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x).view(x.shape[0], -1)
        v2 = F.softmax(self.conv2(v1), dim=-1)
        v3 = v2.view(x.shape[0], x.shape[2], x.shape[3], -1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 256, 256)
