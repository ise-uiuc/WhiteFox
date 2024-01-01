
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(64, 8, (2, 2, 2), stride=1, padding=1, bias=True, dilation=(1, 1, 1))
        self.conv_transpose.weight.data = torch.randn([8, 64, 2, 2, 2])
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return v1
# Inputs to the model
x1 = torch.randn(6, 64, 1, 3, 3)
