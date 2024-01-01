
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2048, 512)
        
    def forward(self, x):
        v = self.linear(x)
        v = v.reshape(1, 512)
        v = torch.nn.functional.relu(v)
        return v
        
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2048)
