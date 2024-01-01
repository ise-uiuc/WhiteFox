
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 3)
 
    def forward(self, x):
        l1 = self.linear(x)
        l2 = l1 * torch.clamp(l1 + 3, 0, 6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(2, 20)
