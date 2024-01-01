
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, groups=3, bias=False, padding_mode='zeros')
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, groups=3, bias=False, padding=1, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        return torch.relu(v1 + v2)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
