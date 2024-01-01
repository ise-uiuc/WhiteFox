
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 3)
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        v1 = v0.permute(0, 2, 1)
        v2 = v0.unsqueeze(1)
        v3 = torch.nn.functional.linear(x0, self.linear1.weight, self.linear1.bias)
        v4 = v0.permute(0, 2, 1)
        v5 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        return v5
# Inputs to the model
x0 = torch.randn(1, 2, 2)
