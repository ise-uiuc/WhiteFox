
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias).permute(0, 2, 1)
        lstm1 = torch.nn.LSTMCell(2, 2)
        v1 = lstm1(v0)
        v2 = v1.permute(0, 2, 1)
        lstm2 = torch.nn.LSTMCell(2, 2)
        v3 = lstm2(v2)
        return v3.permute(0, 2, 1).permute(0, 2, 1)
# Inputs to the model
x0 = torch.randn(1, 3, 2)
