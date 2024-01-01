
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 40)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * 0.01
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model; setting the negative slope to 0 for the first layer of the network
m = Model()
m.linear.bias.data.fill_(0)

# Inputs to the model
x1 = torch.randn(1, 20, 10, 10)
