
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        lstm = torch.nn.LSTM(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v3 = lstm(v1.permute(0, 2, 1))
        return v3[0]
# Inputs to the model
x1 = torch.randn(1, 2, 2)
