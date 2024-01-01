
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 10)

    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias)
        v1 = v0.permute(0, 2, 3, 1)
        lstm1 = torch.nn.LSTMCell(10, 16)
        v2 = lstm1(v1)
        v6 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias)
        v7 = v6.permute(0, 2, 3, 1)
        lstm2 = torch.nn.LSTMCell(10, 8)
        v8 = lstm2(v7)
        return v8.permute(0, 3, 2, 1) + v2 * 2
# Inputs to the model
x0 = torch.randn(1, 3, 10, 10)
x1 = torch.randn(1, 4, 10, 10)
