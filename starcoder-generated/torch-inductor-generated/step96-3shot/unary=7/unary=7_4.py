
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(9, 12)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * torch.clamp(l1+3, min=0, max=6)
        return l2 * 6.0 / 6.0

# Creating input tensor
x1 = torch.randn((1, 9))

# Initializing the model
m = Model()

# Using the model to calculate an output
