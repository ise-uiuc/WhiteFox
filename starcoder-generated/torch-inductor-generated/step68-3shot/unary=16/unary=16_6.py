
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model(64, 64)

# Inputs to the model
x1 = torch.randn(1, 64)
