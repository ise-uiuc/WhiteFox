
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t1 = torch.nn.ConvTranspose1d(35, 91, 2, stride=2, padding=1)
        self.conv_t2 = torch.nn.ConvTranspose1d(12, 23, 4, stride=4, padding=2, output_padding=3)
    def forward(self, x4, x12):
        v2_1 = self.conv_t1(x12)
        v4 = torch.where(v2_1 > -1.958, v2_1, v2_1 * 0.24)
        v2_2 = self.conv_t2(x4)
        v5 = torch.where(v2_2 > 0.763, v2_2, v2_2 * -0.051)
        return v4, v5
# Inputs to the model
x4 = torch.randn(2, 12, 16)
x12 = torch.randn(1, 35, 40)
