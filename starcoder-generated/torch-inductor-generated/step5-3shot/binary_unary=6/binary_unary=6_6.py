
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1 - 0.5
        v3 = torch.nn.functional.ReLU()(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 3)
