
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        lstm = torch.nn.LSTM(2, 2)
        v3 = lstm(v1.permute(1, 0, 2))
        v4 = lstm(v2.permute(1, 0, 2))
        return v3[0] + v4[0]
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
