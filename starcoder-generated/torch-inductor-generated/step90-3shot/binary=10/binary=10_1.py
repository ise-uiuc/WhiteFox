
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 8)
 
    def forward(self, x1):
        v2 = self.linear(x1)*torch.tensor([1.], dtype=torch.float32) + torch.tensor([2.])
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
