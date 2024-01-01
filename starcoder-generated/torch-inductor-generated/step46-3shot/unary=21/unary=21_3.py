
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(2, 2, (3, 3), stride=(1, 1), bias=False, padding=(1, 1))
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.tanh(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 2, 44, 44)
