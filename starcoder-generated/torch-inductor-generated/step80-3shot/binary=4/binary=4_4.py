
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 9)
 
    def forward(self, x1, input2):
        v1 = self.linear(x1)
        v2 = v1 + input2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8)
input2 = torch.randn(2, 9)
