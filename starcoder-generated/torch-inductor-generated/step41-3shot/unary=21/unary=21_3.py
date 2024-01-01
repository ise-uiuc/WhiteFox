
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x):
        y = torch.tanh(self.conv(x))
        return y
# Inputs to the model
x = torch.zeros(1, 1, 2240, 16)
