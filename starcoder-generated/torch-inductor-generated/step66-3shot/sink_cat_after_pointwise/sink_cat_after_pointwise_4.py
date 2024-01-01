
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.randn(1)
        x2 = torch.rand(1)
        x3 = torch.randint(0, 10, (1,))
        return torch.cat([x3, x1, x2], dim=0)
# Inputs to the model
x = torch.randn(2, 3, 4)
