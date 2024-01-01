
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 3)
        self.linear2 = torch.nn.Linear(3, 3)
        self.linear3 = torch.nn.Linear(3, 1)
        self.dropout = torch.nn.Dropout2d()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        v2 = torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias)
        v3 = torch.nn.functional.linear(v2, self.linear3.weight, self.linear3.bias)
        v4 = 42 - v3
        return self.dropout(v4)
# Inputs to the model
x1 = torch.randn(1, 1, 1)
