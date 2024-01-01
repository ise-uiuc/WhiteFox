
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 11, 1)
        self.conv2 = torch.nn.Conv2d(3, 19, 1, padding=1)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = torch.tanh(t1)
        t3 = self.conv2(t2)
        t4 = torch.tanh(t3)
        return t4
# Inputs to the model
x = torch.randn(1, 3, 28, 28)
