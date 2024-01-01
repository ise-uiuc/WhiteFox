
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 1st Conv layer
        self.conv1 = torch.nn.Conv2d(3, 6, 1)
        # 2nd Conv layer
        self.conv2 = torch.nn.Conv2d(6, 16, 3, padding=1)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = torch.tanh(t1)
        y1 = self.conv2(t2)
        return y1
# Inputs to the model
x0 = torch.randn(1, 3, 256, 256)
