
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(343, 257, 1, stride=1, padding=0)
        self.negative_slope = negative_slope
    def forward(self, x4):
        q1 = self.conv_t(x4)
        q2 = q1 > 0
        q3 = q1 * self.negative_slope
        q4 = torch.where(q2, q1, q3)
        return q4
negative_slope = -0.01
# Inputs to the model
x4 = torch.randn(125, 343, 4, 4)
