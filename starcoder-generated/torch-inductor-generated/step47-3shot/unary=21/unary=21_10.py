
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(9, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten the data (n, 3*3*3) to (n, 3*3*3)
        x = F.relu(self.fc1(x))
        return self.tanh(x)
# Inputs to the model
x = torch.randn(1, 9)
