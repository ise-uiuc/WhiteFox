
class Model(torch.nn.Module):
    def __init__(self, input_dim=128, num_heads=8, dropout_p=0.):
        super().__init__()
        self.query = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.key = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.value = torch.nn.Linear(input_dim, input_dim)
        self.softmax_qk = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.output = torch.nn.Linear(input_dim, input_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk / np.sqrt(k.size(-1))
        softmax_qk = self.softmax_qk(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return self.output(output)

# Initializing the model
m = Model(input_dim=256, num_heads=8, dropout_p=0.2)

# Inputs to the model
x = torch.randn(568, 256)
