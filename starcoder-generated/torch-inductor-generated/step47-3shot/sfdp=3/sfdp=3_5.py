
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(10, 32)
        self.proj = torch.nn.Linear(self.emb.embedding_dim, 16)
        self.scale_factor = 4.5
 
    def forward(self, q, k):
        v_emb = self.emb(v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = torch.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v_emb)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 64, 32)
k = torch.randn(8, 138, 32)

scale_factor = 4.5
dropout_p = 0.8

