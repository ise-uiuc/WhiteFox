
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 2)
        self.linear2 = torch.nn.Linear(5, 2)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.linear2(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()
x1 = torch.randn(5)
x2 = torch.randn(5)

## Setting a different values for the weights and biases of the linear transformation layers
m.linear1.weight.data = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
m.linear1.bias.data = torch.tensor([-1, 1])
m.linear1.weight.data = torch.tensor([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
m.linear1.bias.data = torch.tensor([-1, 1])
