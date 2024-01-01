
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 10)
        self.linear2 = torch.nn.Linear(10, 5)
    def forward(self, x3):
        v3 = self.linear1(x3)
        v3 = torch.sigmoid(v3)
        v4 = self.linear2(v3)
        v4 = torch.sigmoid(v4)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5)
