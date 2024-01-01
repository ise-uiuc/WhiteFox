
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(20, 10)

# Inputs to the model
x1 = torch.randn(20, 20)
x2 = torch.randn(20, 20)
# In the example to simplify model conversion, assume that the input tensor of the linear transformation is actually a weight matrix with the size being 20 * 10.
