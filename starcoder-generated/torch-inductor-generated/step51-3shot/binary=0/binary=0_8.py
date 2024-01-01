
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 3, 1, stride=1, padding=1)
    def forward(self, input_tensor, x1, other):
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
input_tensor = torch.randn(1, 8, 64, 64)
x1 = torch.randn(1, 3, 64, 64)
other = torch.randn(v1.shape)
