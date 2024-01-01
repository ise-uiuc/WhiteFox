
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
        self.clamp = torch.nn.Hardtanh(inplace=True)
 
    def forward(self, input):
        h1 = self.linear(input)
        h2 = self.clamp(h1 + 3)
        h3 = h2 * 6
        return h3

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(32, 16)
