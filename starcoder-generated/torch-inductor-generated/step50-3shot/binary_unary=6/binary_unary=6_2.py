
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 64, bias=False)
 
    def forward(self, x1):
        x1 = x1.contiguous()
        t1 = self.fc(x1)
        t2 = t1 - 20.0
        t3 = torch.relu(t2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
