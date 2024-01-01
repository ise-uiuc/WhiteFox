
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        v3 = v1 + x2
        return v3
# Use torch.jit.trace to generate a model proto from input tensors and scripted model code.
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
m_jit = torch.jit.trace(Model(), (x1, x2))
