
class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(28*28, num_classes)
 
    def forward(self, x1):
        v1 = self.linear(x1.view(x1.shape[0], -1))
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model(num_classes=10)

# Inputs to the model
x1 = torch.randn(1, 28*28)
