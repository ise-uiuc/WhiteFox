
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 64)
 
    def forward(self, x1):
        v1 = self.fc(x1) # Apply a linear transformation to x1
        return F.relu(v1) # Apply the ReLU activation function to the output of the linear transformation

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
