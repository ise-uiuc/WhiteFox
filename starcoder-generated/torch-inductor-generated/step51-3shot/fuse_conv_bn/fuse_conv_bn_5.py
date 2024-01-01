
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(1, 1)
    def forward(self, x):
        x = self.l(x)
        for _ in range(6):
            x = self.l(x)
        return x
# Inputs to the model
x = torch.randn(1, 1)
