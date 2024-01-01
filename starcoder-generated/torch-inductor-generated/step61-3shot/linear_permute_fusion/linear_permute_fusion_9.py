
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        lstm = torch.nn.LSTM(2, 2)
    def forward(self, input_):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = lstm(v1.permute(1, 0, 2))
        v3 = torch.nn.functional.linear(v2[0], self.linear.weight, self.linear.bias)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
