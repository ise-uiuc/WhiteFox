
class Model(torch.nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(input_shape, num_classes)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = torch.sigmoid(v7)
        return v8

# Initializing the model
m = Model(128, 10)

# Inputs to the model
x2 = torch.randn(1, 128)
