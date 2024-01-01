
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias)
        v1 = v0.permute(0, 2, 1)
        lstm1 = torch.nn.LSTMCell(4, 2)
        v2 = lstm1(v1)
        return v2.permute(0, 2, 1)
# Inputs to the model
x0 = torch.randn(1, 3, 2)
