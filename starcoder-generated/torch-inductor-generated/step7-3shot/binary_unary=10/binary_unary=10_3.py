
class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, num_classes)
 
    def forward(self, x1):
        x2 = x1.view(x1.size(0), -1)
        v1 = self.fc1(x2)
        v3 = torch.relu(v1)
        return v3

# Initializing the model
m = Model(num_classes)

# Inputs to the model
x1 = torch.randn(15, 1, 28, 28)
