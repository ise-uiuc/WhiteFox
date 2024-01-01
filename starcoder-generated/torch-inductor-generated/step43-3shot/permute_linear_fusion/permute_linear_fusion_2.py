
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softsign = torch.nn.Softsign()
    def forward(self, x1):
        return self.softsign(x1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
