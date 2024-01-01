
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(5, 3, 3, stride=1)
        self.conv_transpose2 = torch.nn.ConvTranspose1d(3, 5, kernel_size=(5, 5), stride=(4, 4))
        self.conv_transpose3 = torch.nn.ConvTranspose1d(5, 13, (3, 3), stride=1)
        self.conv_transpose4 = torch.nn.ConvTranspose1d(13, 1, 5, stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = self.conv_transpose4(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 20)
