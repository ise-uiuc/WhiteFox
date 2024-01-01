
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
        self.conv_b = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv_a(x)
        v2 = torch.transpose(self.conv(v1), 3, 2)
        v3 = torch.transpose(self.conv_b(v2), 3, 2)
        return v3
# Input to the model
x = torch.randn(1, 1, 32, 32)
