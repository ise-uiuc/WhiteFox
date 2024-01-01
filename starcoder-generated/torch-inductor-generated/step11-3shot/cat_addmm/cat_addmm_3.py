
class Model(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.matmul1 = torch.nn.Linear(in_features, 64)
        in_features_for_matmul2 = in_features + 64
        self.matmul2 = torch.nn.Linear(in_features_for_matmul2, out_features)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x1):
        v1 = self.matmul1(x1)
        v2 = self.relu(v1)
        v3 = torch.cat([x1, v2], -1)
        v4 = self.matmul2(v3)
        return v4

# Initializing the model
m = Model(3, 3)

# Inputs to the model
x1 = torch.randn(1, 3)
