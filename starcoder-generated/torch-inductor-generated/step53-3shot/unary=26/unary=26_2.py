
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(336, 230, 3, stride=1, padding=1, bias=True)
    def forward(self, x4):
        p1 = self.conv2d(x4)
        p2 = p1 > 0
        p3 = p1 * 0.775
        p4 = torch.where(p2, p1, p3)
        return torch.quantize_per_tensor(torch.reshape(p4, ( 13, 230,-1)), -2.904, 0, dtype=2)
# Inputs to the model
x4 = torch.randn(3, 336, 15, 10)
