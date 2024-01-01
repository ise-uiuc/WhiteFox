
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv_2 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.tconv_2(x1)
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 256, 256)
