
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.__out_features = negative_slope
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, None)
        v2 = v1 > 0
        v3 = v1 * self.__out_features
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(negative_slope=0.1)

# Inputs to the model
x1 = torch.randn(2, 2, 3)
