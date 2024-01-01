
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
 
    def forward(self, x1):
        a1 = self.linear(x1)
        q_m = torch.baddbmm(self.b, self.u, self.v, transpose_b=True)
        return q_m

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5)
