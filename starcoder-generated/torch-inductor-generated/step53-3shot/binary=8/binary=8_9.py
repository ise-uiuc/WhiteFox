
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 16, bias=True)
        self.linear2 = torch.nn.Linear(16, 8, bias=True)

    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        return v2

# Inputs to the model
x1 = torch.randn(2, 4)
