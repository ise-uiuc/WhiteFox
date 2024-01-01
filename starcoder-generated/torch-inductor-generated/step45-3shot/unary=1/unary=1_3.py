
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, input_tensor):
        x1 = self.linear(input_tensor)
        x2 = x1 * 0.5
        x3 = x1 + (x1 * x1 * x1) * 0.044715
        x5 = torch.tanh(x3)
        x6 = x5 + 1
        x7 = x2 * x6
        return x7

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(2, 16, 5, 5)
