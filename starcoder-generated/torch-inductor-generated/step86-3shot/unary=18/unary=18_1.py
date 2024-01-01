
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(1)
        self.conv1 = torch.nn.Conv2d(1, 1, 3, padding=1)
    def forward(self, x):
        v1 = self.conv1(self.batch_norm(x))
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
