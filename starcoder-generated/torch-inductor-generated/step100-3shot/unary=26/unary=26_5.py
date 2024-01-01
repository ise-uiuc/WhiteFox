
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(95, 95, 4, stride=2, padding=1, output_padding=3, bias=False)
    def forward(self, x8):
        q0 = self.conv_t(x8)
        q1 = q0 > 0
        q2 = q0 * -0.928
        q3 = torch.where(q1, q0, q2)
        return q3
# Inputs to the model
x8 = torch.randn(1, 95, 16, 17)
