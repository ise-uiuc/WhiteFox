
class Model(torch.nn.Module):
    def __init__(self, num_classes, hidden_size, drop_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(drop_out)

    def forward(self, x3, x4):
        v3 = self.linear1(x3)
        v4 = self.dropout(v3)
        v5 = self.linear2(v4)
        v7 = torch.matmul(v5, x4.transpose(-2, -1))
        return v7

# Initializing the model
m = Model(num_classes=16, hidden_size=10, drop_out=0.2)

# Inputs to the model
x3 = torch.randn(2, 10)
x4 = torch.randn(2, 128, 10)
