
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.Linear(1, 1)

    def forward(self, x):
        output = self.ln(x)
        output = relu(output)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1)
