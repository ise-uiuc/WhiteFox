
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Conv2d(other.size(0), other.size(1), 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(other=)

# Inputs to the model
x1 = __prepare_input__()
