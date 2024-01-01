
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, 10)
 
    def forward(self, x1):
        x1 = x1.view(-1, 28 * 28)
        x2 = self.linear(x1)
        x3 = torch.relu(x2)
        x4 = torch.softmax(x3, dim=1)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 28, 28)
