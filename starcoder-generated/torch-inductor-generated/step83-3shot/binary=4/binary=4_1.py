
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        # other is a user placeholder for a tensor thats different from the input tensor
        __add_tensor_name_1 = torch.empty(v1.shape)
        __add_tensor_name_1.uniform_(-1, 1)
        v2 = v1 + __add_tensor_name_1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
