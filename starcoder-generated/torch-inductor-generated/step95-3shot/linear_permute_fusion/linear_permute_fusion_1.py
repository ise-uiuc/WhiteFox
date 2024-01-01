
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.transpose(1, 2)
        lstm1 = torch.nn.LSTMCell(2, 2)
        v3 = lstm1(v2)
        v4 = v3.transpose(1, 2)
        linear1 = torch.nn.Linear(2, 2)
        v5 = linear1(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
