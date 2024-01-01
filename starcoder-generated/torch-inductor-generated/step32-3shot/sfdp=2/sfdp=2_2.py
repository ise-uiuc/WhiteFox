
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 128
        self.num_heads = 8
        self.head_dim = self.embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.embedding = torch.nn.Embedding()
        self.linear1 = torch.nn.Linear()
        self.linear2 = torch.nn.Linear()
 
    def forward(self, x1, x2):
        bz = x1.shape[0]
        q = self.linear1(x1).reshape(bz, self.num_heads, self.head_dim)
        k = self.linear2(x2).reshape(bz, self.num_heads, self.head_dim)
        q = q.permute(..., 1, 0, 2)
        k = k.transpose(1, 0, 2)
        qk = torch.matmul(q, k)
        scaled_qk = qk * self.scaling
        softmax_qk = scaled_qk.softmax(1)
        dropout_qk = softmax_qk *dropout_p
        output = dropout_qk.matmul(v)
        return 
