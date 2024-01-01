
class Model(torch.nn.Module):
    def __init__(self, in_features1, in_features2):
        super().__init__()
        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.fc1 = nn.Linear(in_features1, 4*in_features2)
        self.fc2 = nn.Linear(in_features2, in_features1)
 
    def forward(self, x1, x2):
        v1 = torch.nn.functional.relu(self.fc1(x1))
        v2 = torch.nn.functional.gelu(v1)
        v3 = self.fc2(v2)
        v4 = torch.mm(x2, v3)
        return v4

# Initializing the model
m = Model(__output__.size(1), __output__.size(1))

# Input to the model
x1 = torch.randn(1, __output__.size(1), 4)
x2 = torch.randn(1, __output__.size(1), 4)
