
class Model(torch.nn.Module):
    def __init__(self, in_c, out_c, dim):
        super().__init__()
        self.fc = torch.nn.Linear(in_c, out_c)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
in_c = 8
out_c = 16
dim = 6
m = Model(in_c, out_c, dim)

# Inputs to the model
x1 = torch.ones(1, in_c, dim, dim)
