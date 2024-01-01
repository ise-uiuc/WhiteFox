
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 112)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - torch.tensor([0.299819, 0.587820, 0.114395], dtype=torch.float32).reshape(1, 3, 1, 1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 224, 224, 3)
