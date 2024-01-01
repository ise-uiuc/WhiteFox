
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = torch.nn.Linear(2, 4)
 
    def forward(self, x, add_tensor):
        v1 = self.module(x)
        return v1 + add_tensor

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2)
add_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
