
class Model(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.layers = nn.Linear(1, 9, bias=False)
        self.weight = weight
    def forward(self, x):
        x = self.layers(x)
        x = torch.mm(x, self.weight)
        return x
# Inputs to the model
x = torch.randn(3, 3)
weight = torch.randn((9, 3))
