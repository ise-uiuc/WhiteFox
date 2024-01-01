
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.outconvtransp = torch.nn.ConvTranspose2d(3, 256, kernel_size=(1, 2), padding=(0, 6), stride=2)
    def forward(self, x1):
        v1 = self.outconvtransp(x1)
        v2 = torch.rsqrt(torch.sum(v1 * v1, dim=1, keepdim=True))
        v3 = torch.clamp (v1 * v2, -1, 1)
        v4 = torch.max(v1 * v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 90, 96)
