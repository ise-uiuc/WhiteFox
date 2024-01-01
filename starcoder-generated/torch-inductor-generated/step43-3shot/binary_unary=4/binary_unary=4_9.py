
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, __input_tensor_1__, __input_tensor_2__):
        v1 = self.linear(__input_tensor_1__)
        v2 = v1 + __input_tensor_2__
        v3 = nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10)
x2 = torch.randn(1, 10)
