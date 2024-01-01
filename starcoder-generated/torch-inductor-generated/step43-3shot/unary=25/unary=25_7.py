
def leakyrelu(linear_act_output, negative_slope):
    t1 = linear_act_output
    t2 = t1 > 0
    t3 = t1 * negative_slope
    t4 = torch.where(t2, t1, t3)
    return t4
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)
 
    def forward(self, x1, x2):
      v1 = self.linear(x1)
      v2 = leakyrelu(v1, neg_slope=0.1)
      v3 = leakyrelu(v1, neg_slope=x2)
      return v2, v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(1, 1)

__output__, 