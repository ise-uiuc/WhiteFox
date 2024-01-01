
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2, bias=False)
        self.flatten = torch.flatten
    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = x.matmul(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
