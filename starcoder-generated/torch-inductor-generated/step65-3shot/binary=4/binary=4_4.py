
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 20)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        return v1 + x1 

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 10)
