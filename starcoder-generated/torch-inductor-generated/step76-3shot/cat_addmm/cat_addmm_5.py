
class Model(nn.Module):
    def __init__(self, a=2, b=0):
        super().__init__()
        self.layers = nn.Linear(a,4)
        self.layers2 = nn.Linear(b,4)
    def forward(self, x):
        x = self.layers(x)
        x = self.layers2(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
