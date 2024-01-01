
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 8)
 
    # Note that the output is already processed here
    def forward(self, x1):
        v1 = self.fc(x1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16)
