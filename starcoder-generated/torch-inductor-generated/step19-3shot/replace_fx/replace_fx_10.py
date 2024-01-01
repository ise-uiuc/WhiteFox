
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 2)
    def forward(self, x):
        h = self.fc1(x)
        x = torch.nn.functional.dropout(h)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x)
        return x
# Inputs to the model
x = 1
