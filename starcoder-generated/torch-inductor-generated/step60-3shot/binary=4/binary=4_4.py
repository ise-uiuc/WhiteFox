
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 4)
 
    def forward(self, x):
        v1 = self.linear1(x)
        return v1 + x

# Initializing the model
m = Model()

# Inputs to the model
