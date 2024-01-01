
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        embedding_size = 10
        number_of_classes = 10
        self.linear = torch.nn.Linear(embedding_size, number_of_classes)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
other = torch.randn(1, 10)
