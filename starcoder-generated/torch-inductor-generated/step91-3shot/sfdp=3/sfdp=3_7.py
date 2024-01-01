
class Model(torch.nn.Module):
    def __init__(self, batch, dimension, hidden_dimension, dropout_rate):
        super().__init__()
        self.w_hidden = torch.nn.Linear(dimension, hidden_dimension)
        self.w_out = torch.nn.Linear(hidden_dimension, dimension)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, q, k, v, padding_mask=None):
        x = torch.matmul(q, k.transpose(-2, -1))
        x = self.dropout(x)
        x = self.w_hidden(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = torch.matmul(x, v)
        return x

# Initializing the model
b = 4
d = 16
h = 8
dropout_rate = 0.25
m = Model(b, d, h, dropout_rate)

# Inputs to the model
q = torch.randn(b, d)
k = torch.randn(b, d)
v = torch.randn(b, d)
