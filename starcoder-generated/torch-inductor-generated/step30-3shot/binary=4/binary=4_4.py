
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
 
    def forward(self, input):
        x = self.linear(input)
        x += other
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 128)
