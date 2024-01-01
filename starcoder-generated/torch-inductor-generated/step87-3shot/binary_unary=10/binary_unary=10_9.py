
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(24, 10)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = v2 + other
        v4 = torch.nn.functional.relu(v3)
        return v4
 
 
# Initializing the model
m2 = Model()

# Inputs to the model
x2 = torch.randn(1, 24)

# This is an input for the model. To get the value,
# run this script with the following option,
# `--other 1,2,3,4`
__other__ = # torch.FloatTensor (requires_grad=True)

__output2__ = m2(x2)

