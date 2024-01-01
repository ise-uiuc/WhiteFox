
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Tanh(),
            torch.nn.Tanh(),
            torch.nn.Tanh())
    def forward(self, x1):
        for x in x1:
            x = x + x
            x = x + x
            x = x + x
            x = x + x
            x = x + x
        return self.fc(x)
# Inputs to the model
x1 = torch.randn(4, 4)
