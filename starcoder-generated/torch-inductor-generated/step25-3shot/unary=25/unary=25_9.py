
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(35, 85)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2_t = v1 > 0 # Convert the output of the linear transformation into a Tensor of dtype=torch.bool
        v2 = torch.where(v2_t, v1, v1 * negative_slope)
        return v2

# Initializing model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 85, 35) # The example below requires that the size of the input includes the number of features (i.e., 35) from the previous layer. The size of the input can also include the batch size (i.e., 1), but it is recommended to specify the size of the input without the batch size for the model to work in the real scenarios.
