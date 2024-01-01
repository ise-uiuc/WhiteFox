
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_in, d_out)
        self.linear2 = torch.nn.Linear(d_out, d_out)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.linear2(x2)
        v3 = v1 + v2
        return v3

# Initializing the model
d_in = 10
d_out = 20
m = Model()

# Inputs to the model
x1 = torch.randn(1, d_in)
x2 = torch.randn(1, d_out)
