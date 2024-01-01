
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 256)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1.relu()
        return v2

# Initializing the model
m = Model()

# Inputs to the model
__spec__ = {"inputs":[{"name":"input","shape":[1,6],"dtype":torch.float32}]}
