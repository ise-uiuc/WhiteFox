
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
    def forward(self, x):
        x = F.relu6(self.linear2(x))
        x = F.relu(self.linear1(x))
        x = torch.nn.functional.dropout(x, 0.2)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
