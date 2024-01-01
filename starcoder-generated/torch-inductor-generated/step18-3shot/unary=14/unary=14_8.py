
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose64_no_stride = torch.nn.ConvTranspose2d(1, 3, 1, stride=1, padding=0)
        self.conv_transpose64_stride= torch.nn.ConvTranspose2d(1, 3, 1, stride=2, padding = 0)
    def forward(self,x1, x2):
        v1 = self.conv_transpose64_no_stride(x1)
        v2 = self.conv_transpose64_stride(x2)
        v3 = torch.flatten(x1, 1)
        v4 = torch.flatten(x2, 1)
        v5 = torch.flatten(v1, 1)
        v6 = torch.flatten(v2, 1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 128, 16, 16)
x2 = torch.randn(1, 128, 32, 32)
