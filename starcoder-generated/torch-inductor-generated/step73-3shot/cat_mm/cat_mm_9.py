
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(2, 2)
        self.linear_2 = torch.nn.Linear(2, 2)

    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = self.linear_1(x1)
        v3 = self.linear_2(x2)
        v4 = torch.cat([v1, v2, v3], 1)
        v5 = torch.cat([v4, v4, v4], 1)
        return v5
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
