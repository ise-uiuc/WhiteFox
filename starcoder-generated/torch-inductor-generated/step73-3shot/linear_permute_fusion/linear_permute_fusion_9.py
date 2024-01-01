
class Model5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x3):
        v3 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        v4 = v3.permute(2, 0)
        lstm3 = torch.nn.LSTMCell(4, 3)
        v5 = lstm3(v4)
        return v5
# Inputs to the model
x3 = torch.randn(1, 2, 2)
