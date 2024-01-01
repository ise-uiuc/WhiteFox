
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(2, 4)
    def forward(self, x1):
        v7 = self.linear1(x1)
        v1 = torch.add(v7, self.linear2(x1))
        v4 = v1.permute(0, 2, 1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
