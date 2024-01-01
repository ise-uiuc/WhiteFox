 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(10, 20)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        m1 = torch.nn.functional.relu6(v1 + 3)
        v3 = v1 * torch.clamp(min=0, max=6, m1)
        return v3 / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3,4)
