
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = torch.nn.Linear(size, 10)
 
    def forward(self, x1):
        t1 = self.fc(x1)
        t2 = t1 + x1
        t3 = F.relu(t2)
        return t3

# Initializing the model
m = Model(28 * 28)

# Inputs to the model
x1 = torch.randn(16, 28 * 28)
