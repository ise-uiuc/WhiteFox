
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1.permute(-1, -2, -3)
        v3 = v2.unsqueeze(4)
        v4 = v1.unsqueeze(3)
        v5 = v3 + v4
        v6 = 2 * v5
        v6 = v6.permute(-1, -2, -3, -4, 2)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
