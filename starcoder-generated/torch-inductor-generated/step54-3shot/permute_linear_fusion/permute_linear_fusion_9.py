
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 4, bias=False)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight)
        v4 = torch.nn.functional.linear(v2, self.linear2.weight)
        v5 = torch.nn.functional.relu(v3)
        v6 = torch.nn.functional.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
