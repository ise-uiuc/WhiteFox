
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(10, 10, 1)
        self.conv2 = torch.nn.Conv3d(10, 10, 1, bias=None)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = self.conv2(x).tanh()
        t3 = t1.tanh()
        t3 = t3 + t2
        return t3
# Inputs to the model
x = torch.randn(32, 10, 32, 32, 32)
