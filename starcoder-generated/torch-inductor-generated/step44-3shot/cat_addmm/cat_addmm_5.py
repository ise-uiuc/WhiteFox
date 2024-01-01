
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(10, 10, bias=False)
    def forward(self, x):
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(1, 10)
