
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bmm = torch.nn.Bilinear(4, 6, 5)
    def forward(self, x, y):
        v1 = self.bmm(torch.tanh(x), torch.tanh(y))
        return (v1)
# Inputs to the model
x = torch.randn(64, 4, 64)
y = torch.randn(64, 6, 64)
