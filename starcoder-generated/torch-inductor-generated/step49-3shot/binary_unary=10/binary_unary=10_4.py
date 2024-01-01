
class Model(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim_in, dim_hidden)
        self.linear2 = torch.nn.Linear(dim_hidden, dim_out)
 
    def forward(self, t):
        v = self.linear1(t)
        v = v + t
        v = torch.relu(v)
        v = self.linear2(v)
        return v

# Initializing the model
m = Model(5, 100, 1)

# Inputs to the model
x = torch.randn(1, 5)
