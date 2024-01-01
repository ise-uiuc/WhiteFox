
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = F.relu(v1)
        return v2

# Initializing the model
m = Model(10, 5)

# Inputs to the model
x1 = torch.randn(1, 10)
