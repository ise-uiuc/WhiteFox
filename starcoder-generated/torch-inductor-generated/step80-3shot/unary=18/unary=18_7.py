
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = list(v1.shape)
        v3 = torch.sigmoid(v1)
        v4 = list(v3.shape)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 5, 5)
