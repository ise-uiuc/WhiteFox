
class Model(torch.nn.Module):
   def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

   def forward(self, x, o):
        v1 = self.linear(x)
        v2 = v1 + o
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 10)
o = torch.randn(1)

