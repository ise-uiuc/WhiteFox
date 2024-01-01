
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3, bias=True)
    def forward(self, x):
        x = self.layers(x)
        x = x.flatten(1)
        x = F.softplus(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
