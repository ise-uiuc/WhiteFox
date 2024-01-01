
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + input2
        v3 = v2.relu()
        return v3
# Initializing the model
m = Model()

# Inputs to the model
input2 = torch.randn(1, 2)
x1 = torch.randn(1, 2)
