
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1) 
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 128, 256)
