
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(2, 1, 0)
        lstm1 = torch.nn.LSTM(2, 2)
        v3 = lstm1(v2)
        return v3[0]
# Inputs to the model
x1 = torch.randn(1, 2, 2)
