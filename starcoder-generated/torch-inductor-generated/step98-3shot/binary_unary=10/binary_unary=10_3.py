
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64 * 64 // 4, 64 // 4)
 
    def forward(self, input):
        x = self.linear(input)
        return x + torch.randn(1, 64 // 4)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
