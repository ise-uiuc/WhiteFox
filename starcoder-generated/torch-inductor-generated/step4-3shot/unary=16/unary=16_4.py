
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 56)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        v1 = torch.nn.functional.relu(t1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224)
