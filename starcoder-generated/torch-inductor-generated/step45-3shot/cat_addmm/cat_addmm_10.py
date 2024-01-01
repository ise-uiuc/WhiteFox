
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 4, bias=False)
        self.matmul = torch.matmul
        self.expand = self.layers.expand(3, 3)
    def forward(self, x):
        x = self.layers(x)
        x = self.matmul(self.expand, x)
        return torch.squeeze(x, 1)
# Inputs to the model
x = torch.randn(2, 1)
