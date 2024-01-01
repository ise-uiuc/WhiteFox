
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = torch.mul(2,x)
        x = torch.stack(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
