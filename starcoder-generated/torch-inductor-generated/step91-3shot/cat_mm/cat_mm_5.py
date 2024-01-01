
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(5, 5)
    def forward(self, x):
        return torch.mm(x, self.weight)
# Inputs to the model
x = torch.randn(5, 5)
