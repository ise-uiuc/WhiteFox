
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, input):
        x = self.linear1(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
