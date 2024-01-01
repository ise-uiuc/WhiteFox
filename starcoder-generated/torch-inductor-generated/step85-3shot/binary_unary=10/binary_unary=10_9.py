
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64*64*3, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1.view(len(x1), -1))
        v2 = v1 + torch.tensor(np.random.random(v1.shape), dtype=torch.float32)
        v1 = F.relu(v2)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
