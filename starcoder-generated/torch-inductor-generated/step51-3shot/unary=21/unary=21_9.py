
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 12, 1, stride=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x0):
        x = self.conv_1(x0)
        x = torch.tanh(x)
        return x
# Inputs to the model
x0 = torch.randn(1, 1, 28, 28)
