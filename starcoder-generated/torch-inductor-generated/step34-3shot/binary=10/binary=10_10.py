
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(4, 2)
 
    def forward(self, x1, x2):
        v1 = self.dense(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model()

# Input tensor
x1 = torch.randn(1, 4)

# Other input tensor
x2 = torch.randn(1, 2)
