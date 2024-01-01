
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear1 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.matmul(v2, self.linear1.weight)
        v4 = torch.nn.softmax(v3, 1)
        v2 = torch.matmul(v4, torch.nn.functional.softmax(self.linear1.weight, 0))
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
