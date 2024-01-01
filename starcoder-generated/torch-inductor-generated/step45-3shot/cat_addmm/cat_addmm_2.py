
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.matmul = torch.matmul
    def forward(self, x):
        x = self.layers(x)
        x = x.reshape((3, 2))
        x = self.matmul(x, x)
        return x
# Inputs to the model
x = torch.randn(2, 4)
