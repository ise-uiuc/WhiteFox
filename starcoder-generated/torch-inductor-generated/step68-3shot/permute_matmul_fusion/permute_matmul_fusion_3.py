
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 4)
        self.l2 = nn.Linear(4, 10)
    def forward(self, x1, x2):
        v1 = self.l2(self.l1(x1.permute(0, 2, 1)))
        v2 = torch.matmul(x2.permute(0, 2, 1), x1)
        v3 = torch.matmul(v2, v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
