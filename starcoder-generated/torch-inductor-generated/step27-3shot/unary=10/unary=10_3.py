
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5

# Initializing the model
input_size = 128
hidden_size = 256
m = Model(input_size, hidden_size)

# Inputs to the model
x = torch.randn(32, input_size)
