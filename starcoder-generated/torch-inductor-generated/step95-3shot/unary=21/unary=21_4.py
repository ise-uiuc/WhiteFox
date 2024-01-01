
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu1 = torch.nn.PReLU(2)
    def forward(self, x):
        r1 = self.prelu1(x)
        r2 = torch.tanh(r1)
        return r2
# Inputs to the model
x = torch.randn(1, 2, 3, 4)
