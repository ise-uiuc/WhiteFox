
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        lstm2 = torch.nn.LSTM(1, 1)
        v3 = lstm2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 2)
