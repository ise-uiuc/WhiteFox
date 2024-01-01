
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x0):
        v0 = x0
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        lstm1 = torch.nn.LSTM(4, 2)
        v3 = lstm1(v2)
        lstm2 = torch.nn.LSTM(2, 2)
        v4 = lstm2(v3)
        return v4
# Inputs to the model
x0 = torch.randn(1, 3, 2)
