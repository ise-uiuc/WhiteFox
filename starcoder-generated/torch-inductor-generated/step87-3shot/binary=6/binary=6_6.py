
class Model(torch.nn.Module):
    def __init__(self, linear1_out_features, linear2_in_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, linear1_out_features)
        self.linear2 = torch.nn.Linear(linear2_in_features, 4)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - torch.tensor([0]) # Use tensors to define the other tensor or scalar
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model(8, 8)

# Inputs to the model
x1 = torch.randn(1, 3)
