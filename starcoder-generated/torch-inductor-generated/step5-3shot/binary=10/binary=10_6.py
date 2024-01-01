
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(7 * 7 * 8, 6272)
        self.linear2 = torch.nn.Linear(6272, 1)
 
    def forward(self, x1):
        x1 = x1.view(-1, 7 * 7 * 8)
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(12, 7, 7, 8)

# Generate the tensor "other" and specify it as a keyword argument
x2 = torch.randn(12, 1)
