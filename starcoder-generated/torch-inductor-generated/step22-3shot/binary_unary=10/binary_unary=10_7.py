
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224*224, 512)
 
    def forward(self, x1):
        v1 = input_tensor
        v2 = self.linear(v1) + constant
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224*224)
