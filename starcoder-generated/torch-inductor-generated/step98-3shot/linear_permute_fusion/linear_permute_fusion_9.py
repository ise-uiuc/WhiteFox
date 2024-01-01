
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v0 = x1
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        lstm1 = torch.nn.LSTM(2, 2)
        lstm2 = torch.nn.LSTM(2, 2)
        v4 = lstm1(v2)
        v5 = lstm2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 2)
