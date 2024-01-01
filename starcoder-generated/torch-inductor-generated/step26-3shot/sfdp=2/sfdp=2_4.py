
dropout = 0.1
class Model(torch.nn.Module):
    def __init__(self, h=16, s=False, t=False):
        super().__init__()
        self.h = h
        self.s = s
        self.t = t
        self.num_heads = 2 # The number of attention heads in self-attention
        self.head_dim = h//self.num_heads # The length of each attention head
        self.dropout = dropout # Dropout probability of self-attention
        self.dropout1 = torch.nn.Dropout()
        self.dropout2 = torch.nn.Dropout()
        self.dropout3 = torch.nn.Dropout()
        self.scale_factor = np.sqrt(self.head_dim) if s else 1.0
        if t: 
            self.scale_factor = np.sqrt(self.head_dim)
 
    def _mha(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout1(softmax_qk)
        output = dropout_qk.matmul(value)
        return output
 
    def forward(self, x1, x2):
        x1 = self.dropout1(x1)
        x2 = self.dropout2(x2)
        bsz, tgt_len, embed_dim = x2.shape
        h = self.h
        w = embed_dim
        x2 = x2.mean(dim=2)
        x2 = x2.view(bsz, tgt_len, h)
        v = self._mha(x1, x1, x1)
        w = self._mha(v, x2, x2)
        w = w.view(bsz, tgt_len, h, w).sum(dim=2)
        return w


# Initializing the model
s = True
t = True
h = 128
model = Model(h=h, s=s, t=t)

# Inputs to the model
x1 = torch.randn(1, 16, h)
x2 = torch.randn(1, 64, h)
