
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv2d(8, 4, kernel_size=2, stride=2)
        self.conv2d = torch.ops.aten.conv2d
    def forward(self, x):
        x = self.layers(x)
        t = self.conv2d(x, x, stride=[1, 1])[0]
        t = torch.clamp(t, -100.0, 100.0)
        # t is the same with x here
        return torch.log(t)
# Inputs to the model
x = torch.randn(1, 8, 2, 2)
