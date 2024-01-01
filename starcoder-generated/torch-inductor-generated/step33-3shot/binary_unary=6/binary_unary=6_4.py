
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 16)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - [[5.0], [10.0]]
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
