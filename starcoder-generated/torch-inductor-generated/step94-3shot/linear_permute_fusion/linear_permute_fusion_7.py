
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        lstm1 = torch.nn.LSTMCell(2, 2)
        v2 = lstm1(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 2)
