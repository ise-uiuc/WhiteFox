
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x1, x2, x3):
        v1 = torch.addmm(0.0, x1, x2, self.linear.weight, self.linear.bias)
        v2 = torch.cat([v1, x3], dim=1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
num_features = 16
num_classes = 8

x1 = torch.randn(2, 16)
x2 = torch.randn(16, 8)
x3 = torch.randn(2, 4)
