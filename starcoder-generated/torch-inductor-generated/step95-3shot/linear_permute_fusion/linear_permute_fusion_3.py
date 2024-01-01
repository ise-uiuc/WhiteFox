
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x0, x1):
        v0 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias).permute(0, 2, 1)
        lstm1 = torch.nn.LSTMCell(3, 2)
        v1 = lstm1(v0)
        v2 = v1.transpose(0, 1)
        v3 = v2.transpose(0, 1)
        v4 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias).permute(0, 2, 1)
        lstm2 = torch.nn.LSTMCell(3, 2)
        v5 = lstm2(v4)
        v6 = v5.transpose(0, 1)
        v7 = torch.nn.functional.linear(v6, self.linear.weight, self.linear.bias).permute(0, 2, 1)
        linear1 = torch.nn.Linear(2, 2)
        v8 = linear1(v7)
        return v8.permute(0, 2, 1)
# Inputs to the model
x0 = torch.randn(1, 3, 2)
x1 = torch.randn(1, 3, 2)
