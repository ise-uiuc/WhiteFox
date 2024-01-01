 for testing add on the input
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 32, bias=False)
 
    def forward(self, x, bias):
        t = self.fc(x)
        f = torch.nn.functional.relu
        v1 = f(t+bias)
        return v1

# Initializing the model with default bias
m = Model()

# Generating the input to the model m, with the specified bias
x = torch.randn(1, 16)
b = torch.randn(1, 32)
