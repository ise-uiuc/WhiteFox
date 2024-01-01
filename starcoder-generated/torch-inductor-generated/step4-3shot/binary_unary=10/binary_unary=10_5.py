
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10, bias=True, dtype=torch.float32)
 
    def forward(self, x1):
        t1 = self.linear(x1) 
        t2 = t1 + 1
        t3 = F.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, dtype=torch.float)
