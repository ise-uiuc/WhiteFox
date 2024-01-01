
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.linear.weight = torch.nn.Parameter(([[16.046, 11, 9],
                                                 [14, 10.245, 6],
                                                 [7, 4.0322, 1]]))
        self.linear.bias = torch.nn.Parameter((-9.36, -0.792, 1.215))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other_tensor
        v3 = torch.relu
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 3)
