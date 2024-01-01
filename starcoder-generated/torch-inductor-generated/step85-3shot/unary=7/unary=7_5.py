
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 92)
 
    def forward(self, x):
        l1 = self.linear(x)
        l2 = l1 * torch.clamp(torch.add(l1, 3), 0, 6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 128)
