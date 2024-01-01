
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(42, 1337)
 
    def forward(self, input_tensor, other):
        v1 = self.linear(input_tensor)
        v2 = v1 + other # "other" is a tensor to be added.
        return v2

# Initializing the model
m = Model()

# Input x tensors
x = torch.randn(2, 42)
y = torch.randn(2, 1337)
