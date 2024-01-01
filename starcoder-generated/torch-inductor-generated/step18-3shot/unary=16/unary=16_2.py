
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_in = 512
        num_out = 100
        self.linear = torch.nn.Linear(num_in, num_out)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
