
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
    def forward(self, x0):
        v0 = x0
        v1 = torch.nn.functional.linear(v0, self.linear1.weight, self.linear1.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v2.reshape(1, 8, 1)
        self.linear2 = torch.nn.Linear(1, 2)
        v4 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        v5 = v4.reshape(1, 2, 2)
        return v5
# Inputs to the model
x0 = torch.randn(1, 2, 2)
