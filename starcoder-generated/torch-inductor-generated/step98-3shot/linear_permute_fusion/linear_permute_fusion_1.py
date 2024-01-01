
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v0 = x1
        v2 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v4 = v1.permute(0, 2, 1)
        lstm1 = torch.nn.LSTM(2, 2)
        v5 = lstm1(v2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 2)
