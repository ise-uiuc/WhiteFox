
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.cat((x, x, x), dim=1)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 1)
