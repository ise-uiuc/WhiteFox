
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 8, bias=False)
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        x1 = self.fc(x1)
        if x2 == None:
            x2 = torch.randn(x1.shape)
        x3 = self.conv(x2)
        x1 = x2 + x3
        x4 = self.bn(x1)
        v1 = x1 + x4
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
x2 = torch.randn(1, 3, 292, 292)
