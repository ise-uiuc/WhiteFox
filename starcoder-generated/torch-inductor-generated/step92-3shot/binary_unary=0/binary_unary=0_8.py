
class LinearReluModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_out)
        self.fc2 = torch.nn.Linear(dim_in, dim_out)
    def forward(self, x):
        v1 = self.fc1(x)
        v2 = self.fc2(x)
        v3 = torch.add(v1, v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
dim_in=32
dim_out=32
x = torch.randn(1, dim_in, dim_in)
