
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        l2 = v1 * torch.clamp(torch.min(v1), torch.max(v1), 6)
        return l3


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
