
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0, bias=False)
    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
tensor = torch.randn(1, 3, 128, 128)
