
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(features_in, features_out)
        self.leaky_relu_activation = torch.Tensor(negative_slope)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.leaky_relu_activation
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(0.1)

# Input to the model
x1 = torch.randn(batch_size, features_in)
print(m(x1))

