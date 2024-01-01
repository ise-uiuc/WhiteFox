
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        scale_factor = 1.0 / math.sqrt(512)
        self.q = torch.nn.Linear(512, 384)
        self.k = torch.nn.Linear(512, 384)
        self.value = torch.nn.Linear(512, 384)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.scale_factor = scale_factor
 
    def forward(self, q, k, v):
        q_temp = F.normalize(self.q(q), dim=-1, p=2).div(self.scale_factor)
        k_temp = F.normalize(self.k(k), dim=-1, p=2).div(self.scale_factor)
        v_temp = F.normalize(self.value(v), dim=-1, p=2).div(self.scale_factor)
        qk = torch.matmul(qk, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
dropout_p = ____.rand()
x1 = torch.randn(batch_size, seq_len, 512)
x2 = torch.randn(batch_size, seq_len, 512)
x3 = torch.randn(batch_size, seq_len, 512)
