
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.Tensor([x, x, x])
        return x
# Inputs to the model
x = torch.randn(4, 4)
