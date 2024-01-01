
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 16)
        self.linear2 = torch.nn.Linear(16, 4)
 
    def forward(self, x):
        # Use layer 1 as the model input
        x = self.linear1(x)
 
        # Pass the model input through layer 2
        x = self.linear2(x)
 
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64)
