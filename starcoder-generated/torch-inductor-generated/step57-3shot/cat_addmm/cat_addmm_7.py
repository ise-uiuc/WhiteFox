
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4)
        self.linear2 = nn.Linear(2, 3)
    def forward(self, x, y):
        x = self.linear1(x)
        y = self.linear2(y)
        z = torch.cat((x, y), dim=1)
        return z.flatten(start_dim=1)
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(2, 2)
