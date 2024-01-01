
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        self.dropout = torch.nn.ReLU()
    def forward(self, x1):
        x2 = self.dropout(x1)
        x3 = self.linear(x2)
        y = torch.tanh(x3)
        y = y.permute(0, 2, 1)
        y = self.linear(y)
        x4 = torch.nn.functional.linear(y, self.linear.weight, self.linear.bias)
        return x4
# Inputs to the model
x = torch.randn(7, 2, 2)
