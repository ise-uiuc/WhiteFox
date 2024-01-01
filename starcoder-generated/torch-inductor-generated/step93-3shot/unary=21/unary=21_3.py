
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, 5, padding=2, stride=1)
    def forward(self, x):
        x = torch.tanh(self.conv(x))
        return x
# Inputs to the model
x = torch.randn(1, 3, 485, 485)
