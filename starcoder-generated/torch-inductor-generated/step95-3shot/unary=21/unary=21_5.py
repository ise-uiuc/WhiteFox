
class TestTanh(nn.Module):
    def __init__(self):
        super(TestTanh, self).__init__()
        self.act= nn.Tanh()
    def forward(self, x):
        # v1 = self.bn(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.torch.rand(1, 10, 10, 10)
