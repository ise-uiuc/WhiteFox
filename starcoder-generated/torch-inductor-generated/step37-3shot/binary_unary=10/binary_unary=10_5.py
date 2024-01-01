
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.Linear(3, 8)
 
    def forward(self, x2):
        x3 = self.ln(x2)
        x4 = x3 + x3
        x5 = torch.nn.functional.relu(x4)
        return x5

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)
