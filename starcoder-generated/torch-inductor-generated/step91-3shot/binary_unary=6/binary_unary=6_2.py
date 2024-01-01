
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(10, 19)

# Inputs to the model
x1 = torch.randn(1, 10)
