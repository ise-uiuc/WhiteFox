
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        a = torch.nn.Linear(2, 3)
        self.linear = torch.nn.Sequential(a)
 
    def forward(self, input_tensor, other):
        v1 = self.linear(input_tensor)
        v2 = v1 + other
        output = F.relu(v2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 2)
other = torch.randn(1, 3)
