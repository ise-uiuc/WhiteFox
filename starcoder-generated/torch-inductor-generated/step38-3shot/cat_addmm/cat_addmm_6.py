
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(16, 4)
    def forward(self, x):
        x = self.layers(x)
        shape = x.shape
        x = torch.zeros(*shape, 2)
        x = torch.randn(*shape, 2)
        return x
# Inputs to the model
x = torch.randn(2, 16)
