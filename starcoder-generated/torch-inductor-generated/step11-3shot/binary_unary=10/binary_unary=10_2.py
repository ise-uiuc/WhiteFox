
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1).view(-1, 28 * 28)
        v2 = v1 + torch.zeros(1, 28 * 28, device=x1.device)
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 28 * 28)
