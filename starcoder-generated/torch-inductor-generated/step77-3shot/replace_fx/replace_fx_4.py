

class Model(torch.nn.Module):
    def __init__(self, bias, relu, flatten, dropout):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        t1 = self.relu(x)
        t2 = self.dropout(x)
        return t2
# Inputs to the model
x = torch.randn(1, 2, 2)
