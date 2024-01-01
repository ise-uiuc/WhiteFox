
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=(9, 9), stride=(3, 3), padding=1, dilation=2)
    def forward(self,x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
