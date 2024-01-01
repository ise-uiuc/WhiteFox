
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv_1 = torch.nn.Conv2d(1, 256, 1, padding=0)
        self.conv_2 = torch.nn.Conv2d(256, 256, 3, padding=1, stride=2)
        self.conv_3 = torch.nn.Conv2d(256, 256, 1, padding=0)
    def forward(self, x):
        x = self.conv_1(x)
        x = torch.tanh(x)
        x = self.conv_2(x)
        x = torch.tanh(x)
        x = self.conv_3(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 24, 24)
