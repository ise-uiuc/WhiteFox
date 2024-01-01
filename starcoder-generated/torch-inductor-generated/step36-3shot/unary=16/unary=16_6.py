
class Model(torch.nn.Module): # Define the model class
    
    # Initializing the model class
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 1000)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 64)

# Outputs of the model
