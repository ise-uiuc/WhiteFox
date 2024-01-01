
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3,1)
 
    def forward(self, x1):
        m2 = torch.as_tensor([[2.,20.,30.]])
        v1 = self.linear(x1)
        v2 = v1 - m2
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
