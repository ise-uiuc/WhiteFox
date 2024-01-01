
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 1)
    def forward(self, x1):
        x2 = x1.view(-1, 3) 
        v1 = self.linear1(x2)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 3)
