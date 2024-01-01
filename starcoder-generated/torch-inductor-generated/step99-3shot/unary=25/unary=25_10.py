
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
        self.__output_feature_1_negative_slope__ = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.__output_feature_1_negative_slope__
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(negative_slope=0.2)

# Inputs to the model
x1 = torch.randn(2, 2)
