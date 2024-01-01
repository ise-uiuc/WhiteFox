
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_p=0.5):
        super().__init__()
        self.dropout_p = dropout_p
        self.stack = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.stack.append(nn.MultiheadAttention(input_size, hidden_size, num_heads))
        self.dropout = torch.nn.Dropout(dropout_p)
        self.relu = torch.nn.ReLU()

    def _forward(self, v):
        for attention in self.stack:
            v = self.dropout(attention(v)[0])
            v = self.relu(v)
        return v

    def forward(self, x):
        v = self._forward(x)
        v = self.dropout(v)
        return v

# Initializing the model
m = Model(32, 32, 4)

# Inputs to the model
x1 = torch.randn(2, 64, 32)
