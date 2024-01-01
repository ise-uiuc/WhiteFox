
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv1d(2, 4, 2, groups=2)
    def forward(self, x):
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
