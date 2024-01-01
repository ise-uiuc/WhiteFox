
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.nn.functional.softmax(x2, dim=-1)
        v3 = v3.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        v4 = torch.nn.functional.relu(v4)
        v4 = torch.nn.functional.softmax(v4, dim=-1)
        v5 = x2 * 2
        x4 = torch.mean(x2.to(v2.dtype) * 3, dim=-1)
        v5 = v5 * x4
        v5 = torch.nn.functional.softmax(v5, dim=-1)
        v3 = v4 + v5
        x5 = x3 + x4
        x5 = x5.permute(0, 2, 1)
        v5 = torch.nn.functional.linear(x5, self.linear3.weight, self.linear3.bias)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
