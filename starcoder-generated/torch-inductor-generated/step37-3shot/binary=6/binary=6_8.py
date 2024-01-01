
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 289
        self.hidden_size = 5
        self.layer = torch.nn.Linear(self.input_size, self.hidden_size)
 
    def forward(self, x1):
        v1 = self.layer(x1)
        v2 = v1 - 3
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(1, 289)
