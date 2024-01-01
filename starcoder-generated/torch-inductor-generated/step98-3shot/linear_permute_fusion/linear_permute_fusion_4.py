
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v0 = x1
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        lstm1 = torch.nn.LSTM(2, 2)
        v3 = lstm1(v2)
        v4 = lstm1(v2)
        lstm2 = torch.nn.LSTM(2, 2)
        v5 = lstm2(v3[0])
        v6 = lstm2(v3[1])
        return v3, v4, v5, v6
# Inputs to the model
x1 = torch.randn(2, 4, 3, 2)
