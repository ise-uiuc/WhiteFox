
class Model(torch.nn.Module):
    def __init__(self, seq_len, heads, ff_dim, dropout, scale_factor):
        super().__init__()
        self.query = torch.nn.Linear(seq_len, ff_dim)
        self.key = torch.nn.Linear(seq_len, ff_dim)
        self.value = torch.nn.Linear(seq_len, ff_dim)
        self.dropout_p = dropout
        self.scale_factor = scale_factor
 
    def forward(self, sequence):
        q = self.query(sequence)
        v = self.value(sequence)
        k = self.key(sequence)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(seq_len=300, heads=4, ff_dim=128, dropout=0., scale_factor=4.)

# Inputs to the model
x1 = torch.randn(1, 16, 300)
