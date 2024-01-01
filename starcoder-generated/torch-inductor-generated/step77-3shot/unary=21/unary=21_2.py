
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv_1 = torch.nn.Conv2d(1, 8, 1)
    def forward(self, x2):
        x2 = torch.tanh(self.conv_1(x2))
        return x2
# Inputs to the model
x2 = torch.randn(1, 1, 28, 28)
