
class MyModel(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    def forward(self, x):
        lstm = torch.nn.LSTM(10, 20, 2)
        x, _ = lstm(x)
        return F.dropout(x)

# Inputs to the model
x = torch.randn(1, 3, 10)
