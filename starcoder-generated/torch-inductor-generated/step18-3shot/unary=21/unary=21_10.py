
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        self.tanh_1 = torch.nn.Tanh()
        self.conv_2 = torch.nn.Conv2d(128, 128, (1, 1), stride=(1, 1), padding=(0, 0), groups=128, bias=False)
        self.tanh_2 = torch.nn.Tanh()
    def forward(self, x2):
        x3 = self.conv_1(x2)
        x4 = self.tanh_1(x3)
        x5 = self.conv_2(x4)
        x6 = self.tanh_2(x5)
        return x6, x4, x5
# Inputs to the model
x2 = torch.randn(1, 128, 4, 4)
