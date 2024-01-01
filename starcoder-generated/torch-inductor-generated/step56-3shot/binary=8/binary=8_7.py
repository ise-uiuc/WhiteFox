
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=True, padding_mode='zeros')
    def forward(self, x):
        v = self.conv2d(x)
        return v
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
