
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(206, 107, 10, stride=2, padding=0, output_padding=1, bias=False)
    def forward(self, x3):
        q1 = self.conv_t(x3)
        q2 = q1 > 0
        q3 = q1 * 0.00049
        q4 = torch.where(q2, q1, q3)
        return q4
# Inputs to the model
x3 = torch.randn(11, 206, 15, 455)
