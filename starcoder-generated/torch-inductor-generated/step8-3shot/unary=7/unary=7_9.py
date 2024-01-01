
class Model(torch.nn.Module):
    def __init__(self, input_features_num):
        super().__init__()
        self.linear = torch.nn.Linear(input_features_num, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.clamp(0, 6) + 3 
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model(2)

# Inputs to the model
x1 = torch.randn(1, 2)
