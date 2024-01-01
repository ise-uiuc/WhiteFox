
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.linear1 = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=2)
        self.tanh = torch.nn.Tanh()   
        self.sigmoid1 = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear2a = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v3 = torch.ops.aten.dropout(v3, p=0.10000000000000001, train=False, inplace=False)
        v3 = self.relu(v3)
        v3 = self.softmax(v3)
        v3 = self.tanh(v3)
        w = v3.permute(0, 2, 1)
        return w
# Inputs to the model
x1 = torch.randn(1, 2, 2)
