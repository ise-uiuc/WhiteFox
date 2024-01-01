
class m1(nn.Module):
   def __init__(self):
       super().__init__()
       self.p1 = torch.rand(1)
       self.p2 = torch.nn.Parameter(torch.randn(1))
   def forward(self, x1):
       x2 = torch.nn.functional.relu(self.p1 * x1 + self.p2)
       return x2
# Inputs to the model
x1 = torch.randn(3, 4)
