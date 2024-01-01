
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        x2 = self.relu(x1)
        x3 = torch.nn.functional.dropout(x2, p=0.3, training=True)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
