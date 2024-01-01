
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 9), stride=(1, 1), padding=(1, 1))
        self.conv_transpose = torch.nn.ConvTranspose1d(160, 3, stride=(1, 1), kernel_size=(160, 1), bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 50, 120)
