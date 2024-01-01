
class M2(torch.nn.Module):
    def __init__(self):
        super(M2, self).__init__()
        self.linear = nn.Linear(5, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 8
        return v2

# Initializing the model
mi = M2()

# Inputs to the model
x1 = torch.randn(1, 5)
