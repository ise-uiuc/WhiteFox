
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 1)
 
    def forward(self, x1):
        v1 = torch.reshape(input_tensor, (1, 64 * 64))
        v2 = torch.nn.functional.relu(self.fc1(v1))
        v3 = self.fc2(v2)
        y1 = torch.reshape(v3, (1,))
        return y1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
