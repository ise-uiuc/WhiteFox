
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose_out_channel = torch.nn.ConvTranspose2d(1, 4, 5, stride=1, padding=0)
        self.convtranspose_kernel_size = torch.nn.ConvTranspose2d(1, 4, [20, 120], stride=[20, 1], padding=[100, 10])
        self.convtranspose_stride = torch.nn.ConvTranspose2d(1, 4, 5, stride=[5, 3], padding=0)
        self.convtranspose_pad = torch.nn.ConvTranspose2d(1, 4, 5, stride=1, padding=[[4, 4], [14, 14]])
    def forward(self, x1):
        v1 = self.convtranspose_out_channel(x1)
        v2 = self.convtranspose_kernel_size(x1)
        v3 = self.convtranspose_stride(x1)
        v4 = self.convtranspose_pad(x1)
        v5 = v1 + v2 + v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
