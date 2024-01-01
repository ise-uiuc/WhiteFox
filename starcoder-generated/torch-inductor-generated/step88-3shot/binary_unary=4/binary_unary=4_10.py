s
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, input_tensor, other):
        v1 = self.linear(input_tensor)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)
