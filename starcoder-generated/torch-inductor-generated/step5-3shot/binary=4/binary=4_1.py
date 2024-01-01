
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 16, False)
        self.linear2 = torch.nn.Linear(16, other, False)
 
    def forward(self, input):
        return self.linear2(self.linear1(input)) + self.linear1(input)

# Initializing the model
m = Model(150) # The last layer now has 150 outputs

# Inputs to the model
x1 = torch.randn(1, 10)

# Generating random inputs
x2 = torch.randn(1, 150)
