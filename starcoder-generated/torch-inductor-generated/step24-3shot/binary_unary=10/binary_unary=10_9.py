
class Model(torch.nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, n_classes)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(5, 2)

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 2)
