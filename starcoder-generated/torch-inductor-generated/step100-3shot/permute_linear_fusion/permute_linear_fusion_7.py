
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 1)
        self.linear2 = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v1 = x1.squeeze()
        v2 = v1.permute(1, 0)
        v3 = torch.nn.functional.linear(v2, self.linear1.weight, self.linear1.bias)
        v4 = v3.permute(1, 0)
        v5 = self.linear2(v4)
        v6 = v5.permute(1, 0)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
torch.manual_seed(0)
