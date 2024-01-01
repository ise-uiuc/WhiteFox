
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4*4*3, 7)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - 17
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = model()

# Input to the model
x1 = torch.randn(10, 4*4*3)

