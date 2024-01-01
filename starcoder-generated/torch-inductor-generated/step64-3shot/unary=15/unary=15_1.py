
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1000, 4000, 1, stride=2, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1000, 3, 3)
