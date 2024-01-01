
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 8)
 
    def forward(self, x1):
        x2 = self.l1(x1)
        return x2 + torch.Tensor([1, 2, 3])

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
