
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.tanh = torch.nn.Tanh()
        self.conv_1 = torch.nn.Conv2d(224, 256, 1)
    def forward(self, x):
        r = self.conv_1(x)
        r = self.tanh(r)
        return r
# Inputs to the model
x = torch.randn(1, 224, 7, 7)
