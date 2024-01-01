
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(8, 10)
        torch.nn.init.normal_(self.linear0.weight)
    
    def forward(self, x1):
        v1 = self.linear0(x1)
        v2 = torch.nn.functional.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100, 8)
