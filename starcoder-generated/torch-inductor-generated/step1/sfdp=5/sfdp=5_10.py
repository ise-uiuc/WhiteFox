
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_q = torch.nn.Embedding(q_vocab_size, d_model)
        self.encoder_k = torch.nn.Embedding(k_vocab_size, d_model)
        self.encoder_v = torch.nn.Embedding(v_vocab_size, d_model)
    def forward(self, q_src, k_src, v_src, attn_mask):
        q = self.encoder_q(q_src)
        k = self.encoder_k(k_src)
        v = self.encoder_v(v_src)
        A = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
        A = torch.softmax(A + attn_mask, dims=-1)
        A = torch.nn.Dropout(dropout_p, True)(A)
        O = torch.matmul(A, v)
        return O

# Initializing the model
m = Model()

# Inputs to the model
q_src = torch.randint(0, 10, (4, 6))
k_src = torch.randint(0, 10, (4, 6))
v_src = torch.randn(4, 6, d_model)
attn_mask = torch.randn(4, 6, 6)
