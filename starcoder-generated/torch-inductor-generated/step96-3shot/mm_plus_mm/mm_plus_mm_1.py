
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.tanh(x1 * x2)
        x4 = torch.tanh(x3 * x2)
        x5 = torch.tanh(x4 * x2)
        return x1 + x2 + x3 + x4 + x5
# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(2, 4)
