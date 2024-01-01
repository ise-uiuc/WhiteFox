
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3*64*64, 30)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
 
    def forward(self, x1):
        fc1 = self.maxpool(F.relu(self.fc1(x1)))
        h1 = torch.flatten(fc1, start_dim=1)
        return h1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
