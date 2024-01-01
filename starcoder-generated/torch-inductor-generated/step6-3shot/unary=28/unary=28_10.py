
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 50)
        self.min_value = min
        self.max_value = max
 
    def forward(self, input):
        x = torch.relu(self.fc1(input))
        x = torch.clamp_min(x, self.min_value)
        x = torch.clamp_max(x, self.max_value)
        return x

# Initializing the model
m = Model(0, 1)

# Inputs to the model
input = torch.randn(1, 10)
