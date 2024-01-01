
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = torch.nn.Conv2d(1, 4, (3,3), stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv3x3(x)
        v2 = v1 - 11
        return v2
# Inputs to the model
x = torch.randn(1, 1, 25, 25)
