
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 6)
    def forward(self, x):
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(3, 3)
