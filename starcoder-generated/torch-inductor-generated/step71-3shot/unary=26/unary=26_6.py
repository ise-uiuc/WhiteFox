
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(185, 134, 1, stride=1, padding=0)
    def forward(self, x1):
        r = self.conv_t(x1)
        r1 = r.permute([0,2,3,1])
        r2 = r1 > 0
        r3 = r1 * 0.606
        r4 = torch.where(r2, r1, r3)
        r5 = r4.permute([0,3,1,2])
        return torch.nn.functional.leaky_relu(r5)
# Inputs to the model
x1 = torch.randn(18, 185, 32, 43)
