
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
 
    def forward(self, x1, other_tensor, additional_str='str'):
        v1 = self.linear(x1)
        v2 = v1 + other_tensor
        v3 = F.relu(v2)
        _unused = additional_str
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 4)
other_tensor = torch.randn(8, 8)
