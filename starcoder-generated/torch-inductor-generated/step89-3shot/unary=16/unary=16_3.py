
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 32)
 
    def forward(self, input_tensor):
        x = self.linear(input_tensor)
        out = torch.relu(x)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 256)
