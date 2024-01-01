
batch_size = 64
seq_length = 1
dim = 384
h = 1024
output_dim = 768
num_heads = 64
dropout_p = 0.15

wpe = torch.randn(seq_length, dim)
wte = torch.randn(dim, output_dim)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(dropout_p)
        self.qkv_net = torch.nn.Linear(dim, 3 * dim)
        self.o_net = torch.nn.Linear(dim, output_dim)
        self.dropout2 = torch.nn.Dropout(dropout_p)

    def forward(self, wpe, wte, X):
        X = self.dropout1(X)
        qkv = self.qkv_net(X)
        q, k, v = qkv.chunk(3, dim=-1)
        dots = torch.einsum('sbh,bshd->bhsd', q, k)
        inv_scale_factor = 1.0 / np.sqrt(dim // num_heads)
        scaled_dots = dots * inv_scale_factor
        softmax = scaled_dots.softmax(dim=2)
        dropout = self.dropout2(softmax)
        o = torch.einsum('bhsd,bshd->sbh', dropout, v)
        o = self.o_net(torch.cat((X, o), dim=-1))
        return o

# Initialize a model
m = Model()

# Inputs to the model
