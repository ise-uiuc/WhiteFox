
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 50, bias=True)
 
    def forward(self, input)
        x1 = self.linear(input, other=other_input)
        x2 = self.relu(x1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 10)
other_input = torch.randn(50, 10)
m(input1, other=other_input)

