
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, input_tensor, other=None):
        l1 = self.linear(input_tensor)
        l2 = l1 + other
        l3 = torch.relu(l2)
        return l3

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, 8)
other = torch.randn(1, 8)
