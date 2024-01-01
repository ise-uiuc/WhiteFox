
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.input_channels = 10
        self.hidden_feature_size = 20
        self.negative_slope = negative_slope
        self.linear1 = torch.nn.Linear(self.input_channels, self.hidden_feature_size)
 
    def forward(self, x):
        t1 = self.linear1(x)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(condition)
        return t4

# Initializing the model
m = Model(0.05)

# Input to the model
x = torch.randn(1, 10)
