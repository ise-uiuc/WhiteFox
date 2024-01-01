
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
        self.stack= F.stack
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        x = self.stack((x, x, x), dim=1)
        x = self.cat((x, x), dim=1)
        x = self.cat((x, x, x, x), dim=1)
        x = self.cat((x, x, x, x), dim=0)
        return x
# Inputs to the model
x = Variable(torch.randn(8, 1).float(), requires_grad=True)
