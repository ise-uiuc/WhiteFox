
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.dropout = torch.nn.Dropout()
        self.relu = torch.nn.ReLU()
        self.elu = torch.nn.ELU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.relu(v2)
        v4 = self.dropout(v3)
        v5 = self.elu(v4)
        return v5
# Inputs to the model
x1 = torch.rand(1, 2, 2)
