
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, add_on_tensor):
        v1 = self.linear(x1)
        return v1 + add_on_tensor

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
add_on_tensor = torch.randn(1, 8, 64, 64) 
