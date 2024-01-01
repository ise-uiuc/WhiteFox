
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 1)
 
    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10) # The weight matrix of the affine transformation (fc1.weight)
x2 = torch.randn(1, 1) # The bias vector of the affine transformation (fc1.bias)
