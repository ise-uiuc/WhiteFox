
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
 
    def forward(self, x1):
        b1 = self.linear(x1)
        b2 = b1 * torch.clamp(b1 + 3, 0, 6)
        b3 = b2 / 6
        return b3

# Initializing the model
# from torch.nn.functional import selu as activation_function
m = Model()

# Inputs to the model
x1 = torch.randn(4, 3)
