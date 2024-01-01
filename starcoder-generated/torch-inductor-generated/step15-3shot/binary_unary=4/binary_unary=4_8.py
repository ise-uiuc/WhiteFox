
class Model(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=7, out_features=7)
        self.other_tensor = other_tensor
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other_tensor
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
other_tensor = torch.randn(7)
m = Model(other_tensor)

# Inputs to the model
x1 = torch.randn(1, 7)
