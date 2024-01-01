
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 12, 3, stride=1, padding=0)
    def forward(self, x3):
        v0 = torch.sigmoid(self.conv_t(x3).double())
        v1 = v0 > 0
        v2 = v0 * -0.289
        v3 = torch.where(v1, v0, v2)
        v4 = v0.size(2)
        v5 = v3.int().cpu().numpy()
        v6 = torch.tensor(v5)
        v1 = v4 < v6
        v2 = v0 * -0.603
        v7 = torch.where(v1, v0, v2)
        return v7
# Inputs to the model
x3 = torch.randn(9, 4, 56, 56)
