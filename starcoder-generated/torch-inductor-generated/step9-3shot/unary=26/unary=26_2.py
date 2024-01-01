
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.functional.conv_transpose3d(19, 27, (3, 3, 3))
    def forward(self, x0):
        v1 = self.conv_t(x0)
        v2 = v1 > 0
        v3 = v1 * -0.8689
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x0 = torch.randn(1, 19, 16, 16, 16)
