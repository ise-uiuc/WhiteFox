
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvTranspose2d = torch.nn.ConvTranspose2d(64, 54, 3, stride=1, padding=12, output_padding=12)
    def forward(self, x):
        negative_slope = 0.4764901
        v1 = self.ConvTranspose2d(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.rand(4, 64, 12, 8)
