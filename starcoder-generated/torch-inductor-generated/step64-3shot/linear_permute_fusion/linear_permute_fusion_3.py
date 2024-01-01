
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels=3, stride=1, weight=torch.nn.Parameter(torch.rand((2, 3, 2, 3))), bias=torch.nn.Parameter(torch.rand(2, 6)))
    def forward(self, x1):
        v1 = torch.nn.functional.conv_transpose2d(x1, self.conv.weight, self.conv.bias)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 2)
