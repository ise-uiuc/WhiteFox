
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 5, bias=False, padding=2, dilation=1)
        self.conv2 = torch.nn.Conv2d(5, 1, 5, bias=False, padding=2, dilation=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        return torch.tanh(x)
# Inputs to the model
tensor = torch.randn(1, 1, 65, 65)
