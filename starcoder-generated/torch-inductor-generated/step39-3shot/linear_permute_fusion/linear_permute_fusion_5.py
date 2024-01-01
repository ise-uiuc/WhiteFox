
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.relu(v1)
        v3 = v2.permute(0, 2, 1)
        lstm1 = torch.nn.LSTMCell(2, 2)
        v4 = lstm1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 2)
