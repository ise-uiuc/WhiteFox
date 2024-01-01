
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 3)
        self.linear2 = torch.nn.Linear(3, 1)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = F.sigmoid(v1)
        v3 = self.linear2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1)
