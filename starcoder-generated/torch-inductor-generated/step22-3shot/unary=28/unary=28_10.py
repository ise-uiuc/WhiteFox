
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torch.nn.Linear(5, 2)
        torch.nn.init.normal_(m.weight)
        torch.nn.init.uniform_(m.bias)
        self.linear = m
 
    def forward(self, x):
        t = self.linear(x)
        t1 = torch.clamp_min(t, min=0.01)
        t2 = torch.clamp_max(t1, max=0.1)
        return t2

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(1, 5)
