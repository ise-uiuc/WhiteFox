
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 10)

    def forward(self, x):
        x = x + 3
        x = torch.clamp(x, min=0)
        x = torch.clamp(x, max=6)
        x = self.linear(x)
        return x


# Initializing the model
m = Model()
m.to(DEVICE)

# Inputs to the model
x = torch.randn(1, 1, device=DEVICE)

y = m(x)
print(y)
