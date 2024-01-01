
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(6, 10)
        self.layers1 = nn.Linear(10, 5)
    def forward(self, x, y):
        x = self.layers(x)
        y = self.layers1(y)
        x = torch.cat((x,x), dim=0)
        y = torch.cat((y,y), dim=0)
        z = torch.cat((x,y), dim=1)
        return z
# Inputs to the model
x = torch.randn(3, 6)
y = torch.randn(3, 5)
