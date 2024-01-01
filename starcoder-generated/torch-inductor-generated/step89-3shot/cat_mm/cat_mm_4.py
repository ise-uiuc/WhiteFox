
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(2, 3, 5, requires_grad=True)
    def forward(self, x):
        return torch.cat([self.weight, self.weight], 2)
# Inputs to the model
x = torch.randn(1, 3, 2)
