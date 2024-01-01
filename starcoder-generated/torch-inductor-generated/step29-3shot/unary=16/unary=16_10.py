
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 2)
 
    def forward(self, x0):
        v1 = self.linear(x0)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                  dtype=torch.float32, device=torch.device('cpu'), requires_grad=True)
