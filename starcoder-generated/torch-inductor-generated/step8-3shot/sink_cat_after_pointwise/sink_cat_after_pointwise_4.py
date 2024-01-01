
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 6)
        self.weight2 = torch.randn(6, 2, 4)
    def forward(self, x):
        z = self.weight.view(-1, 6)
        y = z.tanh()
        x = y.view(-1, 2, 4)
        x = torch.cat([x, x], dim=0)
        x = x.add(self.weight2).tanh()
        y = torch.relu(x)
        y = x[-1]
        y = y.permute(1, 0)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
