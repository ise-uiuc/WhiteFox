
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose1d(64, 17, kernel_size=(2), stride=(3), bias=False, padding=(5))
        self.non_linear = torch.nn.ReLU(inplace=True)
    def forward(self, x3):
        v1 = self.conv_t(x3)
        v2 = self.non_linear(v1)
        return v2
# Inputs to the model
x3 = torch.randn(10, 64, 100)
