
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_1 = torch.nn.ConvTranspose1d(32, 16, 17, stride=1, padding=0, bias=False)
        self.conv_t_2 = torch.nn.ConvTranspose1d(16, 1, 17, stride=1, padding=0, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        x2 = self.conv_t_1(x1)
        x3 = self.conv_t_2(x2)
        x4 = self.sigmoid(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 32, 1000)
