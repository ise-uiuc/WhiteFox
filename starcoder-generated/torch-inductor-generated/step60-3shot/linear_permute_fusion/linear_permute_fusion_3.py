
class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x2):
        v2 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 2, 1)
        v4 = v3.flatten(0, 1)
        lstm3 = torch.nn.LSTMCell(4, 3)
        v5 = lstm3(v4)
        return v5
# Inputs to the model
x2 = torch.randn(1, 2, 2)
