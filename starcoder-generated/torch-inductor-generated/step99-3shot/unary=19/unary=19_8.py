
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(192, 2)

    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = torch.sigmoid(v1)
        return v2

m = Model()

# Inputs to the model
x1 = torch.randn(1, 192)
