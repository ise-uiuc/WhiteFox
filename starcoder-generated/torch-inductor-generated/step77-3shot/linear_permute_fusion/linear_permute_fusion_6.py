
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x3):
        lstm2 = torch.nn.LSTM(2, 2)
        v1 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v2.permute(1, 0, 2)
        v4 = v3.permute(1, 0)
        return lstm2(v4)
# Inputs to the model
x3 = torch.randn(1, 2, 2)
