
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, (1, 5))
        self.tanh = torch.nn.Tanh()
    def forward(self, input):
        x = self.conv(input)
        x = self.tanh(x)
        return x
# Inputs to the model
input = torch.randn(1, 1, 3, 3)
