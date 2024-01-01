
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 4, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3 

# Initializing the model
m = Model()

# Inputs to the model, other is a parameter with a Tensor value
x1 = torch.randn(4, 6)
