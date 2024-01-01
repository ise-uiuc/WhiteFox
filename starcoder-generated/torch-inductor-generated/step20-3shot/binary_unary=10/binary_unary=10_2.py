
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2):
        o1 = torch.nn.functional.linear(x1, x2[0], x2[1], bias=x2[2])
        o2 = torch.relu(o1)
        return o2

# Initializing the model
m = Model()

# Inputs to the model
in_data1 = torch.rand(1, 7)
in_data2 = (torch.rand(1, 7), torch.rand(7), torch.rand(1))
