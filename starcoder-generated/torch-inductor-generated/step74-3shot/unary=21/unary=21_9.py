
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.gelu = torch.nn.GELU()
    def forward(self, x):
        v1 = self.conv1(x)
        t1 = torch.tanh(v1)
        v2 = self.conv2(t1)
        t2 = self.gelu(v2)
        return t2
# Inputs to the model
x = torch.randn(1, 1, 128, 128)
