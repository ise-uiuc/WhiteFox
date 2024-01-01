
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.linear2 = torch.nn.Linear(3, 3)
        self.linear3 = torch.nn.Linear(3, 1)
    def forward(self, x0):
        v1 = x0.permute(0, 2, 3, 1)
        v2 = torch.relu(torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias))
        v3 = torch.abs(torch.nn.functional.linear(v1, self.linear2.weight, self.linear2.bias))
        v4 = torch.nn.functional.linear(v1, self.linear3.weight, self.linear3.bias)
        return v4
# Inputs to the model
x0 = torch.randn(1, 10, 10, 3)
