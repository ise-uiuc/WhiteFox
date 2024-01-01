
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.zeros(3)
        for i in range(3):
            x[i] = torch.sum(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
