
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x2):
        v2 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 2, 1)
        lstm2 = torch.nn.LSTM(2, 2)
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        v5 = lstm2(v4.permute(1, 0, 2))
        return v5[0]
# Inputs to the model
x2 = torch.randn(1, 3, 2)
