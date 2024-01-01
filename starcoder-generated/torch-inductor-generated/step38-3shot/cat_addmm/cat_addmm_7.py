
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 20)
    def forward(self, x):
        x = self.layers(x)
        s = x.mean(0)
        s = s.sum()
        return s
# Inputs to the model
x = torch.randn(1, 1)
