
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64*64*3, 512)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        l2 = v1 > 0
        l3 = v1 * 0.1
        v4 = torch.where(l2, v1, l3)
        return v4

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(1, 64*64*3)
