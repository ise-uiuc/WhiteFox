
class Model(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.linear = torch.nn.Linear(36, 10)
        self.other_tensor = other_tensor
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + self.other_tensor
        v3 = F.relu(v2)
        return v3

# Initializing the model
other_tensor = torch.randn(1, 36)
m = Model(other_tensor=other_tensor)

# Input to the model
x = torch.randn(1, 36)
