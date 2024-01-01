
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 6)
    def forward(self, x):
        z = self.weight.view(-1, 6)
        x = z.tanh()
        x = torch.cat([x, x], dim=1)
        y = torch.relu(x)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
