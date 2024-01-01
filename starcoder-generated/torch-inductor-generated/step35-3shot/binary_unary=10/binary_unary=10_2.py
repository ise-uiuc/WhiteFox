
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(25088, 1000)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + input_1
        v3 = torch.relu(v2)
        return v3
m = Model()

# Inputs to the model
x1 = torch.randn(1, 25088)
input_1 = torch.randn(1, 1000)
