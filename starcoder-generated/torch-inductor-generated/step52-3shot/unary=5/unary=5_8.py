
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(7, 20, 1, stride=1, padding=0)
    def forward(self, x0):
        v0 = self.conv_transpose(x0)
        v1 = v0 * 0.21449275362394012
        v2 = v0 * 0.4933822846162827
        v3 = v0 * 0.5077301052518364
        v4 = torch.erf(v0)
        v5 = v4 + 1
        v6 = v3 * v5
        return v6
# Inputs to the model
x0 = torch.randn(8, 7, 35, 35)
