
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        v0 = x1 # Permute the shape of the input tensor
        x2 = self.fc(v0)
        v1 = F.relu(x2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
