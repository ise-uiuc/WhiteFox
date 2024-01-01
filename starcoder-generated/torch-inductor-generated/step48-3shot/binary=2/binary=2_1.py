
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_before = torch.nn.Conv3d(32, 58, kernel_size=(1, 5, 5), stride=(1, 1, 1), bias=False, padding=(0, 0, 0))
        self.conv_after = torch.nn.Conv3d(58, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), bias=False, padding=(0, 0, 0))
    def forward(self, x2):
        v1 = self.conv_before(x2)
        v2 = v1 - (-3.4)
        v3 = self.conv_after(v2)
        v4 = v3 + (3.4)
        return -v4
# Inputs to the model
x2 = torch.randn(4, 32, 1000, 2000, 2000)
