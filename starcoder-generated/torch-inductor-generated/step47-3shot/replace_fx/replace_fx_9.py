
class model(torch.nn.Module):
    def __init__(self):
	    super().__init__()
	    self.relu = torch.nn.ReLU()
	    self.lstm = torch.nn.LSTM(220, 512, 2, bidirectional=True)
	    self.linear = torch.nn.Linear(512 * 2, 4)
    def forward(self, x1, x2):
	    a = x1 + x2
	    b = torch.abs(a)
	    c = self.relu(b)
	    d, e = self.lstm(c)
	    f = self.linear(d)
	    return torch.nn.functional.dropout(f, p=0.75)
# Inputs to the model
x1 = torch.randn(1, 2, 220)
x2 = torch.randn(1, 4, 220)
