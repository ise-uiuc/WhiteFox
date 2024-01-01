
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = torch.nn.Linear(1792, 7)
 
    def forward(self, x1, x2):
        v1 = self.linear2(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1792)
x2 = torch.randn(1, 7)
