
class TorchModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2=None, x3=None):
        y1 = torch.tanh(x1)
        y2 = torch.cat((y1, x1), dim=-1)
        return y2
# Inputs to the model
x1 = torch.randn(5, 2, 8)
