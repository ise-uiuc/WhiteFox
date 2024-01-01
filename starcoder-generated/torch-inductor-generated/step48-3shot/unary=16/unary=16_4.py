
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)
 
    def forward(input):
        v1 = self.linear(x1)
        v2 = v1.relu()
        return v2

# Inputs to the model
x1 = torch.randn(1, 3)
