
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1024)
        self.transpose = torch.nn.Transpose()
        self.linear1 = torch.nn.Linear(2048, 1)
    def forward(self, x1):
        linear1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        linear2 = self.transpose(linear1, 1, 0)
        linear3 = torch.nn.functional.linear(linear2, self.linear1.weight, self.linear1.bias)
        return linear3
# Inputs to the model
x1 = torch.randn(1, 1, 3072)
