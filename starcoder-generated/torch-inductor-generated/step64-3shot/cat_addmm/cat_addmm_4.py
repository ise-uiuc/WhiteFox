
class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.op = torch.add
    def forward(self, x, y, z):
        x = self.op(x, y)
        x = self.op(x, z)
        return x
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(2, 2)
z = torch.randn(2, 2)
