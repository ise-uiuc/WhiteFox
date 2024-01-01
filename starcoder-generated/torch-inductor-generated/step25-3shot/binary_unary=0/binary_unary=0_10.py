
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 5, stride=1, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 4, 224, 224)
