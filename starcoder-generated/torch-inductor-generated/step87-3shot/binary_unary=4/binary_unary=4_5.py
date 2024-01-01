
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

other = torch.randn(16)
