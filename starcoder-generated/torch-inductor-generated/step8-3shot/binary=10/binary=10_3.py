
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(56, 128)
 
    def forward(self, x):
        v1 = self.layer1(x)
        v2 = v1 + other_tensor
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 56)
