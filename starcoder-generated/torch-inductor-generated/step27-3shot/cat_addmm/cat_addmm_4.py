
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Linear(2, 3)
        self.layers2 = nn.Linear(4, 2)
        self.layers3 = nn.Linear(6, 8)
    def forward(self, x):
        x = self.layers1(x)
        t1 = torch.cat((x, x))
        t2 = torch.cat((t1, t1))
        x = self.layers2(t2)
        x = self.layers3(x)
        return x
# Inputs to the model
x = torch.randn(2,2)
