
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_relu = torch.nn.Linear(10, hidden_size).relu
        
    def forward(self, x1):
        v1 = self.linear_relu(x1)
        v2 = v1 + x1
        return v2

# Initializing the model
m = Model(hidden_size)

# Inputs to the model
x1 = torch.randn(10)
