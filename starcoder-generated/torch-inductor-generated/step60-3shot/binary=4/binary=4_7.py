
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 512)
        self.other = torch.nn.Parameter(torch.tensor([1.0, 2.0]).unsqueeze(1))
 
    def forward(self, x0):
        v1 = self.linear(x0)
        v4 = v1 + self.other
        return v4

# Initialize the model
m = Model()

# Inputs to the model
x0 = [torch.randn(1, 256), torch.randn(1, 256)]
