
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = torch.nn.Parameter(torch.Tensor([2.0]))
    def forward(self, x):
        y = torch.cat((self.b.repeat(()), x), dim=1)
        y = torch.tanh(y)
        y = torch.tanh(y)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
