
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 5)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.Tensor([[0.9, 0.6, 0.3, 0.75, 0.25]])
