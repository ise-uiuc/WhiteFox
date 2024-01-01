
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 3)
 
    def forward(self, x1):
        m = torch.empty(x1.shape[0], 2, 3, dtype=torch.float32, device=x1.device)
        x = self.fc(x1)
        torch.addmm(m, x, x)
        x = torch.cat([x, m], dim = 2)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
