
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n1, n2, n3 = 2048, 1024, 256
        self.linear1 = torch.nn.Linear(n1, n2)
        self.linear2 = torch.nn.Linear(n2, n3)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.linear2(v3)
        return v4

# Initializing the parameters
torch.manual_seed(0)
m = Model()

# Inputs to the model
x1 = torch.randn(128, 2048)
