
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)
    def forward(self, x):
        x = x + torch.ones(2, 2)
        x = x * 2
        x = self.layer(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
