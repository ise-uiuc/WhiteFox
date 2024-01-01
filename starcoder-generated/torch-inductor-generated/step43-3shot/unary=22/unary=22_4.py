
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer = torch.nn.Linear(input_size, output_size)
 
    def forward(self, x1):
        v1 = self.layer(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model(input_size=300, output_size=150)

# Inputs to the model
x1 = torch.randn(1, 300)
