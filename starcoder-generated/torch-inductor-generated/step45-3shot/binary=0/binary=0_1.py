
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 11, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(11, 9, 3, stride=1, padding=1)
    def forward(self, x1, other=False):
        v1 = self.conv1(x1)
        if other == False:
            other = torch.randn(v1.shape)
        v2 = v1 + 0.1
        v3 = v2 + 0.1
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
