
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nhead = 2  # head count
        self.head_dim = 32  # dimension of each head
        self.n_queries = 8  # number of queries
        self.n_keys = 4  # number of keys
        self.n_values = 4  # number of values
        self.qkv = torch.nn.Linear(32, 32 * 3)  # linear transformation of 32-dimensional queries, keys and values
        self.out = torch.nn.Linear(32, 32)  # linear transformation between the representations
        self.scale = self.head_dim ** -0.5  # scaling factor for softmax

    def forward(self, x1, x2):
        qkv = self.qkv(x1)  # Compute a 3-dimensional tensor
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)  # divide qkv into q, k and v respectively
        q = q.unsqueeze(1).repeat(1, self.n_queries, 1)  # repeat q along its batch axis and n_queries times
        k = k.unsqueeze(1).repeat(1, self.n_keys, 1)  # repeat k along its batch axis and n_keys times
        v = v.unsqueeze(1).repeat(1, self.n_values, 1)  # repeat k along its batch axis and n_values times
        att = torch.bmm(q.permute(0, 2, 1), k)  # Compute the matrix multiplication of q and k
        inv_scale = 1.0 / np.sqrt(self.head_dim)
        scaled_att = inv_scale * att
        weights = scaled_att.softmax(dim=-1)  # Compute the softmax of the matrix multiplication of q and k
        o = torch.bmm(weights, v)  # Compute the matrix multiplication of attention weights between key and value
        o = o.flatten(0, 1)  # Flatten the o
        o = self.out(o)  # Compute the out
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 32, 3)  # queries
x2 = torch.randn(2, 32, 3)  # keys
