
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 2, stride=2, padding=1)
        self.dropout = torch.nn.Dropout2d(p=0.4)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.dropout(v1)
        v3 = self.conv2(v2)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.6
max = 0.9
# Inputs to the model
x1 = torch.randn(1, 3, 7, 7)
