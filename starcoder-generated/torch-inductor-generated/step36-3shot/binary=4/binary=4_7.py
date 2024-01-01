
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1) # Transformation applied to x1
        v2 = v1 + x2 # Other tensor added to output of transformation applied to x1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16)
