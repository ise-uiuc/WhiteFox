
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1, bias=False)
        self.conv1 = torch.nn.Conv2d(32, 8, 3, stride=1, padding=1, bias=False)
        self.conv_a = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1, bias=False)
        self.conv_b = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1, bias=False)
    def forward(self, x):
        v1 = self.conv0(x)
        v2 = self.conv1(x)
        v3 = self.conv_b(v1)
        v4 = self.conv_a(v2) 
        return v4 + v3
# Inputs to the model
x = torch.randn(1, 8, 32, 32)
