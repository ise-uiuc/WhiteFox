
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
     
        self.bias = nn.Parameter(torch.zeros(10))
        self.mask = torch.nn.Parameter(torch.ones(10)*0.1)

    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.add(v1, self.bias.unsqueeze(0).expand(x1.size(0), 10), alpha=self.mask)
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
