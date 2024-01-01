
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_block = torch.nn.Sequential(torch.nn.ConvTranspose2d(3, 32, 2, padding=0, stride=2), torch.nn.ReLU(inplace=False))
    def forward(self, x):
        v1 = self.conv_transpose_block(x)
        return v1
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
