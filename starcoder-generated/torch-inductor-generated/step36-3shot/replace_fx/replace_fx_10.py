
class Model(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p)
    def forward(self, x):
        h1 = self.relu(x)
        h2 = h1 * h1
        h3 = self.dropout(h2)
        h4 = h1 / h3
        return h4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
