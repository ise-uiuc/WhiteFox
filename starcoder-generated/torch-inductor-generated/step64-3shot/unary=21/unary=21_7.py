
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv_1 = torch.nn.Conv2d(1, 16, 1)
        self.conv_2 = torch.nn.Conv2d(16, 1, 1)
    def forward(self, x):
        x = x.float()
        h = nn.Hardtanh()
        x = self.conv_1(x)
        x = h(x)
        x = self.conv_2(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
