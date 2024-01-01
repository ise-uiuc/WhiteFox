
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28*28, 10)
 
    def forward(self, input):
        var_1 = self.linear(input)
        var_2 = torch.sigmoid(var_1)
        var_3 = var_1 * var_2
        return var_3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 28*28)
