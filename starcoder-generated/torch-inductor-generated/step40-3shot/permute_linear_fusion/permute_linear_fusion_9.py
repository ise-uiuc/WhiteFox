
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
        self.linear3 = torch.nn.Linear(1, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = x2.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        v5 = torch.nn.functional.relu(v4)
        v6 = v5.permute(0, 2, 1)
        v7 = torch.nn.functional.linear(v6, self.linear3.weight, self.linear3.bias)
        v8 = torch.nn.functional.softmax(v7, dim=-1)
        return v8
# Inputs to the model
x1 = torch.randn(1, 2, 2)
