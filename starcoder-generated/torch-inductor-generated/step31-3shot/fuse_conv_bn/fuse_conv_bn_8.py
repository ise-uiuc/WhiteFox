
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.activation = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(3)
        self.linear2 = torch.nn.Linear(3, 3)
    def forward(self, x0):
        y0 = self.linear(x0)
        y1 = self.activation(y0)
        y1 = self.bn(y1)
        y1 = self.linear2(y1)
        return y1
# Inputs to the model
x0 = torch.randn(1, 3)
