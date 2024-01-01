
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 16)
 
    def forward(self, x9, min_value=0.2, max_value=0.8):
        return torch.clamp(torch.clamp(torch.nn.functional.relu(self.fc(x9)), min_value), max_value)

# Initializing the model
m = Model()

# Inputs to the model
x9 = torch.randn(1, 3)
