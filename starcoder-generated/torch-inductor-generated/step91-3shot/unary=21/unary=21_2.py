
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 7, 5, stride=5, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 7, 5, stride=5, padding=1)
    def forward(self, x2):
        v1 = self.conv1(x2)
        v2 = self.conv2(x2)
        v3 = torch.tanh(v1)
        return v3 * v2
# Inputs to the model
x2 = torch.randn(1, 3, 32, 32)
