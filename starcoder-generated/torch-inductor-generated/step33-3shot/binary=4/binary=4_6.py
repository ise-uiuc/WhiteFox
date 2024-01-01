
class Model(torch.nn.Module):
   def __init__(self):
       super().__init__()
       self.linear1 = torch.addmm(
            torch.nn.Linear(20, 20),
            torch.nn.Linear(20, 20),
            torch.nn.Linear(20, 20),
        )

   def forward(self, x0, x1, x2):
       v0 = self.linear1(x0)
       v1 = torch.linear(x1)
       v2 = torch.linear(x2)
       return v0 + v1, v0 + v2

# Initializing the model
m = Model()

# Inputs to the model

x0 = torch.randn(10, 20)
x1 = torch.randn(10, 20)
x2 = torch.randn(10, 20)

__output__, __output_1__ = m(x0, x1, x2)

