
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_out = torch.nn.Linear(3, 1)        
 
    def forward(self, x1, x2, x3, x4):
        y = self.linear_out(x1.clone())
        t1_0 = x2 + 1.0 * y
        t2_0 = y + x3
        t3_0 = x3 + 1.0 * y
        t4_0 = x1 + 1.0 * y
        s = torch.cat([t1_0, t2_0, t3_0, t4_0], dim=1)
        return s

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
x3 = torch.randn(1, 3)
x4 = torch.randn(1, 3)
