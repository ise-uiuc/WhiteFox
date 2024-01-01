
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 4)
 
    def forward(self, x2, other=None):
        v1 = self.linear(x2)
        if other is not None:
            v1 += other.squeeze()
        v2 = torch.nn.functional.relu(v1, inplace=False)
        return v2

# Initializing the model
m = Model()

# Input tensor
x_tensor1 = torch.randn(1, 10)
x_tensor2 = torch.randn(1, 10)
# Inputs to the model
