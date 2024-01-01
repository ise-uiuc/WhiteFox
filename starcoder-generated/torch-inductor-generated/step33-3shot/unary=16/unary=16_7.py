
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1, False)
    def forward(self, inputs):
        v = self.linear(inputs)
        v = F.relu(v)
        return v
	
# Inputs to the model
inputs = torch.randn(2, 4)

# Initializing the model
m = Model()

