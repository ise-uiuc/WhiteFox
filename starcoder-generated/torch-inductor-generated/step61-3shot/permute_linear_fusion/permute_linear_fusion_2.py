
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = x1.permute(0, 2, 1)
        x3 = x2 * self.linear1.weight
        x4 = x3.permute(2, 1, 0)
        x5 = x4 * self.linear2.weight
        x6 = x5.permute(1, 2, 0)
        return x6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
