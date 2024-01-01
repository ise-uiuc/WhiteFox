
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, x):
        v1 = self.linear(input)
        return v1[:, :128]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
