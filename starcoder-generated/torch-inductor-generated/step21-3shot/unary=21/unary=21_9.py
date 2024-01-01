
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 2, 3, stride=1, padding=1)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = self.conv2(t1)
        tanh1 = torch.tanh(t2)
        return tanh1
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
