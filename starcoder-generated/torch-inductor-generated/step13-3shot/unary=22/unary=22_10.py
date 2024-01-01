
def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        torch.nn.init.zeros_(m.bias)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
        self.linear.apply(init_weights)

    def forward(self, x1):
        v0 = x1.detach().clone()
        v0.requires_grad_(True)
        v1 = self.linear(v0)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
