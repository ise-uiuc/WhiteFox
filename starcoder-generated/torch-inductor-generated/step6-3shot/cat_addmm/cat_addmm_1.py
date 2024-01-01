
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        output_dim = 4 * 2
        self.fc = torch.nn.Linear(2, output_dim)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.addmm(x1, v1, v1.transpose(1, 0))
        v3 = torch.cat([x1, v2, v1.transpose(1, 0)], 1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
