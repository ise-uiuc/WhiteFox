
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 5, 3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(5, momentum=0.05)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.batch_norm(x1)
        return x1
# Inputs to the model
x = torch.randn(2, 64, 2, 2)
