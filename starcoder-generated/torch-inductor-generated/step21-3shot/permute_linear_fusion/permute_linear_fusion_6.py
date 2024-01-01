
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x2 = torch.nn.functional.relu(v2)
        z2 = x2 * 5 - 2
        z3 = torch.nn.functional.softmax(z2, dim=0) * 2
        return z2, z3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
