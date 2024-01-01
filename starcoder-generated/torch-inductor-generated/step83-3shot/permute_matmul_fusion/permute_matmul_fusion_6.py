
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = self.linear(x2)
        v0 = x1.permute(0, 2, 1)
        v3 = torch.bmm(v0, v1)[0][1][0]
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
