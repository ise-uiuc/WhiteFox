
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(64, 10)

    def forward(self, x):
        y = self.l1(x)
        y = torch.sigmoid(y)
        return y

# Initializing the model
m = Model()

# Initialize optimizer
opt = torch.optim.Adam(m.parameters())

# Inputs to the model
x = torch.randn(2, 64)
label = torch.randint(10, [2])

for _ in range(20000):
    # Zero gradient buffers
    grads = [ 0.0 ] * len(m.parameters())
    # Forward pass
    y = m(x)
    # Compute loss
    e = -sum(label * torch.log(y) + [ -e for e in (1 - label) * torch.log(1 - y) ])
    # Compute gradients
    e0 = -sum( (label / y) + (1 - label) / (1 - y))
    e0.backward()
    opt.step()

