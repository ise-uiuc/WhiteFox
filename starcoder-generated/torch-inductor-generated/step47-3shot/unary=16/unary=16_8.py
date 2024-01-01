
class Model(torch.nn.Module):
    def __init__(self, input_channels, input_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size**2 * input_channels, input_size**2)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
x = torch.randn(1, 3, 64, 64)
y = x.view(x.size(0), -1)
m = Model(input_channels=3, input_size=64)

# Inputs to the model
