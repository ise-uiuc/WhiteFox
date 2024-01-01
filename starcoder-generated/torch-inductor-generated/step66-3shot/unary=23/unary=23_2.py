
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(8, 3, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = v2.mean(3).mean(3).mean(3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 8, 12, 13, 15)
