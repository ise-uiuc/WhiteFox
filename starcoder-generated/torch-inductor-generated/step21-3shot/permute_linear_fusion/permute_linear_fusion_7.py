
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        x3 = torch.mean(v2, dim=-1)
        v3 = x3.transpose(1, 2)
        v4 = self.linear2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
