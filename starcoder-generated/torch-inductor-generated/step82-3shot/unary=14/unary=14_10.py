
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose3d(3, 64, (3, 5, 4), (2, 5, 5), (1, 3, 1), 0, 1, 1)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = v2.contiguous()
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 4, 4, 5)
