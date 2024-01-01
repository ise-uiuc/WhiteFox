
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(10, 5, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(5, 2, kernel_size=1)
    def forward(self, x4):
        t1 = self.conv1(x4)
        t2 = torch.tanh(t1)
        t3 = self.conv2(t2)
        t4 = torch.tanh(t3)
        t5 = self.conv3(t4)
        t6 = torch.tanh(t5)
        return t6
# Inputs to the model
x4 = torch.randn(1, 3, 64, 64)
