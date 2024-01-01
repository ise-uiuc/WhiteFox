
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers1(x)
        x = torch.sin(x)
        x = self.layers1(x)
        x = torch.cat((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
