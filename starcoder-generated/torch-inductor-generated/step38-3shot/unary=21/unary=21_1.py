
class ModelTanh(torch.nn.Module):
    def __init__():
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, kernel_size=2, stride=2)
    def forward(self, x):
        v1 = x
        v2 = torch.tanh(v1)
        t1 = x
        self.conv(t1)
        t1 = torch.tanh(t1)
        t2 = self.conv(t1)
        t1 = t1 + t2
        v3 = torch.tanh(t1)
        v4 = self.conv(v1)
        v5 = v1 + v4
        v6 = torch.tanh(v5)
        v5 = v5 + v6
        return v5
# Inputs to the model
x = torch.randn(10, 3, 128, 128)
