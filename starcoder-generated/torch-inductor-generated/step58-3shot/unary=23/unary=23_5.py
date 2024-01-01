
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(1,1, kernel_size=3, stride=1, padding=2, bias=True)
    def forward(self, x4):
        v1 = self.conv_transpose(x4)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x4 = torch.randn(1,1,27,2,17)
