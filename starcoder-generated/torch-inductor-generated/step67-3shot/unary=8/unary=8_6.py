
class Model_0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.randn(8, 16, 3, 1, 1, 1)
        self.conv_transpose1 = torch.nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.conv_transpose1.weight.data = weight
        self.conv_transpose1.bias = torch.nn.Parameter(torch.randn(8))
    def forward(self, x1, x2):
        p1 = self.conv_transpose1(x1)
        v1 = x2 * p1
        return v1
# Inputs to the model
x1 = torch.randn(1, 16, 34, 19, 19)
x2 = torch.randn(1, 8, 34, 19, 19)
