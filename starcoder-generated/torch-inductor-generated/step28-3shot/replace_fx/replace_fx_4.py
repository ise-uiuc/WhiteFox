
class MyModel(torch.nn.Module):
    def __init__(self,):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(8,2)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.dropout(x)
        x = torch.rand_like(x)
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.dropout(x)
        x = torch.nn.functional.dropout(x)
        return x[0]
