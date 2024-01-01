
class Model(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.w_query = torch.nn.Linear(embedding_dim, embedding_dim)
        self.w_query.weight = torch.nn.Parameter(torch.eye(embedding_dim))
        self.w_key = torch.nn.Linear(embedding_dim, embedding_dim)
        self.w_key_b = torch.nn.Linear(embedding_dim, embedding_dim)
        self.w_value = torch.nn.Linear(embedding_dim, embedding_dim)
        self.w_value_b = torch.nn.Linear(embedding_dim, embedding_dim)
        self.bias1 = torch.nn.Parameter(torch.zeros(1, 1, 1, embedding_dim))
        self.bias2 = torch.nn.Parameter(torch.zeros(1, 1, 1, embedding_dim))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        b, n, h = x.size()
        q = self.w_query(x).view(b, n, 1, h)
        k = self.w_key(x).view(b, n, 1, h)
        k_b = self.w_key_b(x).view(b, 1, n, h)
        k = torch.cat((k, k_b), 1)
        v = self.w_value(x).view(b, n, 1, h)
        v_b = self.w_value_b(x).view(b, 1, n, h)
        v = torch.cat((v, v_b), 1)
        qk = torch.matmul(q, k)
        scaled_qk = qk.mul(1.0 / math.sqrt(h)).add(self.bias1)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v).view(b, n, h)
        output = output + self.bias2
        return output

# Initializing the model with a dummy embedding_dim
m = Model(13)

# Inputs to the model
x = torch.randn(1, 10, 13)
