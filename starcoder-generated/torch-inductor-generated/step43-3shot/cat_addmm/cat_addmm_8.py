
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x
# Inputs to the model
x = torch.randn(2, 2)
