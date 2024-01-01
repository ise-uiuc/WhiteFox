
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = torch.nn.functional.interpolate(v2, None, 1, 'nearest')
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
