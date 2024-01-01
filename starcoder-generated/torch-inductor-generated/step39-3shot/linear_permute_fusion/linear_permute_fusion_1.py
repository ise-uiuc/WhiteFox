
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        lstm1 = torch.nn.LSTMCell(2, 2)
        v4 = lstm1(v2)
        lstm2 = torch.nn.LSTMCell(2, 2)
        v5 = lstm2(v4)
        return v1 + v4 + v5
# Inputs to the model
x1 = torch.randn(3, 2, 2)
x2 = torch.randn(3, 2, 2)
