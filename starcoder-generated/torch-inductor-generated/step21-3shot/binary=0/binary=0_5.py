
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 2, (3, 2), stride=2, padding=0)
        self.batch_norm = torch.nn.BatchNorm2d(2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.batch_norm(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 224, 224)
