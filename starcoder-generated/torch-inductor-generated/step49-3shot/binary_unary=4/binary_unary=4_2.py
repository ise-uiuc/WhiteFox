
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, t1=None):
        return torch.relu(self.linear(x1) + t1)

# Initializing the model
m = Model()

# Input tensor to the model
x1 = torch.randn(1, 3, 64, 64)
# Other input tensor to the model
t1 = torch.randn(1, 8)
