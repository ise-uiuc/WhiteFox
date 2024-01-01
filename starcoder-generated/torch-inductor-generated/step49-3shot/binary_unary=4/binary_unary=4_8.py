
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, __input_tensor, __other):
        v1 = self.linear(__input_tensor)
        v2 = v1 + __other
        v3 = torch.nn.functional.relu(v2)
        return v3


# Initializing the model
m = Model()

# Inputs to the model
__input_tensor = torch.randn(1, 3)
__other = torch.randn(1, 8)
