
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(32 * 32 * 3, 2048)
        self.linear2 = torch.nn.Linear(2048, 10)
 
    def forward(self, x1):
        v1 = self.linear1(x1.view(-1, 32 * 32 * 3))
        v2 = v1 + x1
        v3 = F.relu(v2)
        return self.linear2(v3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
