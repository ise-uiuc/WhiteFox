
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8 * 32 * 32, 1000)
        self.linear2 = torch.nn.Linear(1000, 10)
 
    def forward(self, x1):
        v1 = x1
        v2 = torch.reshape(v1, [v1.shape[0], -1])
        v3 = self.linear1(v2)
        v4 = v3 + self.linear2.bias
        return v4

# Initializing the model
m = Model()

# Initialize parameters
m.linear1.weight.data.fill_(3.14)
m.linear1.bias.data.fill_(100)
m.linear2.weight.data.fill_(123.45)
m.linear2.bias.data.fill_(60)

# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
