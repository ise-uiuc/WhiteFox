
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x):
        v2 = torch.tanh(self.conv(x))
        return v2
# Inputs to the model
x = torch.zeros(1, 1, 3, 3)
