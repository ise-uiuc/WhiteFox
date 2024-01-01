
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 64)
 
    def forward(self, t1):
        t2 = self.linear(t1)
        t3 = F.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
t1 = torch.randn(512)
