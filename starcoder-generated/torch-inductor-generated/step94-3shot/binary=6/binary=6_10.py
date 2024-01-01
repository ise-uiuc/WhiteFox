
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__in_features = 64
        self.__out_features = 64
        # 'inplace' and 'device' can be directly assigned default values here
        self.linear = torch.nn.Linear(self.__in_features, self.__out_features)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
