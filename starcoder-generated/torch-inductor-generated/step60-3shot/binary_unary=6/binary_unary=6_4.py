
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 32)
 
    def forward(self, x1):
        q = self.linear(x1)
        q = q - 9
        a = torch.nn.functional.relu(q)
        return a

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
