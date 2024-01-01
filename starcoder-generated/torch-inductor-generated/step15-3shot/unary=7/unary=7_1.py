
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 100) # Apply linear transformation with output size 100 to the input tensor
 
    def forward(self, x):
        v1 = self.fc1(x)
        v2 = v1 * torch.clamp(v1 + 3, min=0, max=6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(10, 784)
