
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = 0.123456789 + torch.tanh(x1)
        x3 = x2.permute(0, 2, 1)
        x4 = torch.nn.functional.linear(x3, self.linear1.weight, self.linear1.bias)
        x4 = torch.tanh(x4)
        x5 = torch.nn.functional.linear(x4, self.linear2.weight, self.linear2.bias)
        y = x5.reshape(1, 4, 1)
        return y
# Inputs to the model
x1 = torch.randn(1, 2, 2)
