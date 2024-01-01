
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.line1 = torch.nn.Linear(64*64*3, 100)
        self.line2 = torch.nn.Linear(100, 10)
 
    def forward(self, x1):
        v1 = self.line1(x1.view(1, -1))
        v2 = F.relu(v1)
        v3 = self.line2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1,3,64,64)
