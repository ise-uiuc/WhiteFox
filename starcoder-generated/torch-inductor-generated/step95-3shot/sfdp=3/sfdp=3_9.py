
class Model(torch.nn.Module):
    def __init__(self, input1_size, input2_size, hidden_size, dropout_p):
        super().__init__()
        self.attention = torch.nn.Linear(input1_size * input2_size, hidden_size)
        self.dropout_qk = torch.nn.Dropout(dropout_p)

    def forward(self, q, k, v):
        q_k = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = torch.sqrt(torch.tensor(float(v.size(-1))/float(q_k.size(-1))))
        q_k_scaled = q_k.mul(scale_factor)
        softmax_qk = q_k_scaled.softmax(dim=-1)
        dropout_qk = self.dropout_qk(softmax_qk)
        o = dropout_qk.matmul(v)
        return o

# Initializing the model
m = Model(input1_size=3, input2_size=3, hidden_size=6, dropout_p=0.5)

# Inputs to the model
q = torch.randn(1, 3, 4)
k = torch.randn(1, 3, 8)
v = torch.randn(1, 8, 4)
