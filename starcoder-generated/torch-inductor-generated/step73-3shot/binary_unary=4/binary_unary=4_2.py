
class Model(torch.nn.Module):
    def __init__(self, input_size=256, hidden_size=32):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
 
    def forward(self, x1, *, t):
        v1 = self.linear(x1)
        v2 = v1 + t
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
t = torch.randn(1, 32)
