
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
input_shape = [1, 128]
x1 = torch.randn(input_shape)
original = torch.randn([1, 1])
