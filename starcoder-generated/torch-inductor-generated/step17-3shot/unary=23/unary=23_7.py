
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1d = torch.nn.ConvTranspose1d(3, 6, 3, stride=4, padding=5)
        self.conv_transpose2d = torch.nn.ConvTranspose2d(5, 1, 3, stride=1, padding=1)
        self.conv_transpose3d = torch.nn.ConvTranspose3d(7, 6, (1, 4, 3), stride=(1, 2, 3), padding=(2, 1, 4))
    def forward(self, x1):
        v1 = torch.tanh(self.conv_transpose1d(x1))
        v2 = torch.tanh(self.conv_transpose2d(v1))
        v3 = torch.tanh(self.conv_transpose3d(v2))
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 127)
