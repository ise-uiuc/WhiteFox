
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 30)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2, inplace=True)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)

# Initialization of the tensor to which another tensor should be added
x2 = torch.randn(1, 3)
