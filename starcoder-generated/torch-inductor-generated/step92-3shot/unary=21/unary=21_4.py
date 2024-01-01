
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(64, 1, 3, stride=(2, 2), groups=2)
    def forward(self, x):
        t1 = self.conv2(x)
        t2 = torch.tanh(t1)
        return t2
# Inputs to the model
x = torch.randn(1, 64, 32, 32)
