
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 16, 3, stride=1, padding=1)
        # Modify the kernel weight tensor of self.conv1
        self.conv.weight.data = torch.randn(16, 5, 3, 3).type(torch.float32)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 32, 32)
