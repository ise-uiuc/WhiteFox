
class Model(torch.nn.Module):
    def __init__(self):
        inv_scale_factor = 1.0 / math.sqrt(8)
        super().__init__()
        self.query = torch.nn.Linear(8, 8)
        self.key = torch.nn.Linear(8, 8)
        self.value = torch.nn.Linear(8, 8)
        self.inv_scale_factor = inv_scale_factor
        self.dropout_p = 0.3
 
    def forward(self, x):
      x1 = self.query(x)
      x2 = self.key(x)
      x3 = self.value(x)
      z1 = torch.matmul(x1, x2.transpose(-2, -1))
      z2 = z1.div(self.inv_scale_factor)
      z3 = torch.nn.functional.softmax(z2, dim=-1)
      z4 = torch.nn.functional.dropout(z3, p=self.dropout_p)
      z5 = torch.matmul(z4, x3)
      return z5

# Inputs to the model
x = torch.randn(8, 8)
