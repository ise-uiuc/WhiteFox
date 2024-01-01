
class Model(torch.nn.Module):
    def __init__(self, query_size=20, key_size=12, value_size=5, head_num=2, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.head_num = head_num
        self.dropout_qk = torch.nn.Dropout(self.dropout_p)
        self.matmul_qk = torch.nn.Linear(query_size + key_size, head_num)
 
    def forward(self, query, key, value, inv_scale_factor):
        batch_size = query.shape[0]
        qk = self.matmul_qk(torch.cat([query, key], dim=-1))
        qk = qk.reshape(batch_size, -1, self.head_num).transpose(dim0=-2, dim1=-1)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout_qk(softmax_qk)
        output = dropout_qk.matmul(value.reshape(batch_size, -1, self.head_num).transpose(dim0=-2, dim1=-1))
        return output.reshape(batch_size, -1)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 20, 64)
key = torch.randn(8, 12, 64)
value = torch.randn(8, 5, 64)
inv_scale_factor = torch.randn(1)
