
class Model(torch.nn.Module):
    def __init__(self, batch_size, d_model, num_heads):
        super().__init__()
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
 
    def forward(self, query, key, value, scale_factor=1, dropout_p=0):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output
 
model = Model(batch_size=1, d_model=3, num_heads=1)
query = torch.randn(1, 1, 3)
key = torch.randn(1, 1, 3)
value = torch.randn(1, 1, 3)
dropout_p = 0
scale_factor = 1
