
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 256)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + v2
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m1 = Model()
m2 = Model()
 
# Initialize input tensors
x1 = torch.randn(16, 512)
