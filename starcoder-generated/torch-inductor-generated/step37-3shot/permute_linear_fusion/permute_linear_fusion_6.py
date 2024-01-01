
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.nn.functional.linear(v2, self.linear2.weight, self.linear2.bias)
        v4 = torch.nn.functional.relu(v3)
        v4 = v4.permute(0, 2, 1)
        v3 = v3.to(torch.float16)
        return torch.sum(torch.abs(v3 - v4), dim=(-1))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
