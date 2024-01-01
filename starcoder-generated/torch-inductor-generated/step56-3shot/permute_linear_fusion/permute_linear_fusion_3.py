
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear1 = torch.nn.Linear(1, 1)
    def forward(self, x1):
        x2 = torch.nn.functional.relu(self.linear1(1))
        v1 = self.linear(x1)
        x2 = torch.nn.functional.relu(x2)
        v2 = torch.randn(1)
        v3 = v1.permute(0, 2, 1).index_select(2, v4)
        v3 = torch.take(v1, v4).detach()
        v3 = x2 * x2 + v2 + 1
        v3 = x2 / x2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
