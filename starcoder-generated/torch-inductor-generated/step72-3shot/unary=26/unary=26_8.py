
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(6, 5, 2)
    def forward(self, x5):
        q1 = self.conv_t(x5)
        q2 = q1 > 0
        q3 = q1 * -0.68977776
        q4 = torch.where(q2, q1, q3)
        return q1.reshape(-1, 1)
# Inputs to the model
x5 = torch.randn(91, 6)
