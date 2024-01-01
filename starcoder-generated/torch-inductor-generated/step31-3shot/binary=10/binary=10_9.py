
class Model(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.other_tensor = other_tensor
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 + self.other_tensor
        return t2

# Initializing the model
other_tensor = torch.randn(8, 3)
m = Model(other_tensor)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
