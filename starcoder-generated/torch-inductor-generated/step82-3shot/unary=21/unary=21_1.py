
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(2, 4, 3, stride=2, padding=0)
    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
tensor = torch.randn(1, 2, 16, 16)
