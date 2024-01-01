
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(n, m)
        self.linear1 = Linear(m, m)
        self.linear2 = Linear(m, m)
 
    def forward(self, input, other):
        x = self.linear(input)
        x = x + other
        x = self.linear1(x)
        x = x + other
        x = self.linear2(x)
        return x

# Initializing the model
m = Model(n, m)

# Input to the model
input = torch.randn(1, 23)
other = torch.randn(30, 23)
