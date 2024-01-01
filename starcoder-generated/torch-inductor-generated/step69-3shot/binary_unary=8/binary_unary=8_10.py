
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(512, 1024, 2, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.quantize_per_tensor(v1, float(1), int(0), torch.quint8)
        return v2
# Inputs to the model
x1 = torch.randn(1, 512, 32, 32)
