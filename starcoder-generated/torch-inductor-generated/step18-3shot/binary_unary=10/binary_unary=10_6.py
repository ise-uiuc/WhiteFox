
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 512)
 
    def forward(self, input):
        x = input
        x = self.fc(x)
        x += torch.rand_like(x) * 0.01
        result = torch.nn.functional.relu(x)
        return result

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
