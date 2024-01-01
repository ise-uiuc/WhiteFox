
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 8, bias=True)
        self.clamp = torch.nn.Threshold
 
 
    def forward(self, inputs):
        intermediate = self.linear(inputs)
        output = self.clamp(intermediate, 0, 6)
        output = output / 6
        return output

# Initializing the model
m1 = Model()

# Inputs to the model
inputs = torch.randn(1, 1)
