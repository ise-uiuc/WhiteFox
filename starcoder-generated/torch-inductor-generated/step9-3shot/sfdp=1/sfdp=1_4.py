
class Model(torch.nn.Module):
    def __init__(self, num_heads, embedding_size, dropout_p=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.head_size = embedding_size // num_heads
        self.dropout_p = dropout_p
        self.q_layer = torch.nn.Linear(embedding_size, embedding_size, bias=False)
        self.k_layer = torch.nn.Linear(embedding_size, embedding_size, bias=False)
        self.v_layer = torch.nn.Linear(embedding_size, embedding_size, bias=False)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1, x2):
        q = self.q_layer(x1)
        k = self.k_layer(x2)
        v = self.v_layer(x2)
        q = torch.reshape(q, (-1, q.shape[1], self.num_heads, self.head_size))
        q = torch.transpose(q, 1, 2)
        k = torch.reshape(k, (-1, k.shape[1], self.num_heads, self.head_size))
        k = torch.transpose(k, 1, 2)
        v = torch.reshape(v, (-1, v.shape[1], self.num_heads, self.head_size))
        v = torch.transpose(v, 1, 2)
        dropout_qk = torch.nn.functional.dropout(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.shape[-1]), p=self.dropout_p)
        output = torch.matmul(dropout_qk, v)
        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (-1, output.shape[1], output.shape[2] * output.shape[3]))
        return output

# Initializing the model
m = Model(5, 30, 0.1)

# Inputs to the model
x1 = torch.randn(5, 10, 30)
x2 = torch.randn(5, 20, 30)
