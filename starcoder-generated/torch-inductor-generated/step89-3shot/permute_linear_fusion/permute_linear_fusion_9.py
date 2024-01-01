
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 2)
        self.dropout1 = torch.nn.Dropout(0.)
        self.dropout2 = torch.nn.Dropout(0.)
        self.dropout3 = torch.nn.Dropout(0.)
        self.bn1 = torch.nn.Identity()
        self.bn2 = torch.nn.Identity()
        self.bn3 = torch.nn.Identity()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v2 = self.bn1(v2)
        v2 = torch.nn.functional.relu(v2)
        v2 = self.linear2(v2)
        v2 = self.bn2(v2)
        v2 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = self.dropout1(v2)
        v2 = v2 + v4
        x2 = v2 * x1
        x3 = v2 * x1
        v3 = self.linear3(x3)
        v3 = self.bn3(v3)
        v3 = torch.nn.functional.softmax(v3, dim=-1)
        v5 = torch.mean(v3, dim=-1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
