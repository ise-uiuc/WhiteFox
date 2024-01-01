
class mymodel(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.linear2 = torch.nn.Linear(2, 2)
        self.drop1 = torch.nn.Dropout(0.2)
        self.linear1 = torch.nn.Linear(2, num_labels)
    def forward(self, x):
        x = x + x
        x = self.linear2(x)
        x = torch.nn.functional.dropout(x, 0.2)
        x = self.linear1(x)
        x = torch.nn.functional.dropout(x, 0.2)
        return torch.nn.functional.relu6(x)
# Inputs to the model
x = torch.randn(1, 2)
