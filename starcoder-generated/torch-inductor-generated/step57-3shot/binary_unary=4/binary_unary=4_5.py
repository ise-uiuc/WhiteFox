
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1):
        v0 = torch.zeros([], dtype=torch.long) # Index to be used with torch.unsqueeze to add an empty batch dimension
        x2 = torch.unsqueeze(x1, v0)
        v1 = self.linear(x2)
        v2 = v1 + x1
        v3 = torch.nn.functional.relu(v2)
        return torch.squeeze(v3)
 
# Initializing the model
m = Model()
 
# Input to the model
x1 = torch.randn(5)
 
# Calling the model
