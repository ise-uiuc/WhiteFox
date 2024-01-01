
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias).permute(0, 2, 1)
        lstm1 = torch.nn.LSTMCell(2, 2)
        v2 = lstm1(v1)
        linear1 = torch.nn.Linear(2, 2)
        v3 = linear1(v2)
        return v3.permute(0, 2, 1).permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
