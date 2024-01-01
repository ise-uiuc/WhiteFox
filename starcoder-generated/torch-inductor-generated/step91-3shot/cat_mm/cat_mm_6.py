
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(3):
            setattr(self, 'layer' + str(i + 1), torch.nn.Linear(1, 1, bias=False))
    def forward(self, x1, x2):
        v1 = self.layer1(x1)
        return torch.cat([v1, v1, v1], 0)
# Inputs to the model
x1 = torch.randn(3, 1)
x2 = torch.randn(1, 1)
