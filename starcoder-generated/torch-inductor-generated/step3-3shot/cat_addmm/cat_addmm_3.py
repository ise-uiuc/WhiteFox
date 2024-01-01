
class Model(torch.nn.Module):
    def __init__():
        super().__init__()
        self.fc = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        b1 = torch.cat([x1, x1, x1], dim=1)
        v1 = self.fc(b1)
        v3 = torch.cat([v1, v1, v1], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
