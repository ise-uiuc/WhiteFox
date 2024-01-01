
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convT = torch.nn.ConvTranspose2d(62, 47, 12, stride=2, padding=1, bias=False)
    def forward(self, x1):
        p1 = self.convT(x1)
        p2 = p1 > 0
        p3 = p1 * -0.239
        p4 = torch.where(p2, p1, p3)
        return p4
# Inputs to the model
x1 = torch.randn(1, 62, 11, 78)
