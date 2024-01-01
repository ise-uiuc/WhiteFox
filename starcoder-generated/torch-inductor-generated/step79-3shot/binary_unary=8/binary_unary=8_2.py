
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 7, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        t1 = v1 + v1
        v2 = torch.relu(t1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
