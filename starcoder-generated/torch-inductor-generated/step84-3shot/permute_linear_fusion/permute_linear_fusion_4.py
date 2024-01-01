
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = self.softmax(v1)
        v4 = torch.permute(v2, 1, 0)
        v5 = torch.nn.functional.linear(v4, self.linear2.weight, self.linear2.bias)
        v7 = self.softmax(v5)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
