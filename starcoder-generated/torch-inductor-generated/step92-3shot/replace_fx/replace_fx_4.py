
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
    def forward(self, x):
        x = self.conv(x)
        torch.dropout(x, 0.2)
        # call any model
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
