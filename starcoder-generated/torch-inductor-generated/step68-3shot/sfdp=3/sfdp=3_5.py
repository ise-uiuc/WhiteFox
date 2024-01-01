
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qdim = 4
        self.kdim = 5
        self.vdim = 3
        self.numheads = 2
        self.query = torch.nn.Linear(self.qdim, 8 * self.numheads)
        self.key = torch.nn.Linear(self.kdim, 8 * self.numheads)
        self.value = torch.nn.Linear(self.vdim, 8 * self.numheads)
        self.dropout_p = 0.6
 
    def forward(self, q1, k1, v1):
        query = self.query(q1)
        key = self.key(k1)
        value = self.value(v1)
        qdim, _ = q1.size()
        kdim, _ = k1.size()
        vdim, _ = v1.size()
        q = query.reshape(qdim, self.numheads, -1).transpose(1, 0)
        k = key.reshape(kdim, self.numheads, -1).transpose(1, 0)
        v = value.reshape(vdim, self.numheads, -1).transpose(1, 0)
        dropout_p = self.dropout_p
        scale_factor = (kdim*kdim) ** -0.5
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
q1 = torch.randn(2, 4)
k1 = torch.randn(3, 5)
v1 = torch.randn(3, 3)
