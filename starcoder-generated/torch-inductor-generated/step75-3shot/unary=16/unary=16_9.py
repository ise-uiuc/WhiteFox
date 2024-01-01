
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(
            self.linear = torch.nn.Linear(input_size = 5, output_size = 3),
        )
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.nn.functional.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 5)
