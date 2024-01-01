
class Model(torch.nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2
 
# Initializing the model
m = Model(2, 10)

# Inputs to the model
x1 = torch.randn(1, 10)
