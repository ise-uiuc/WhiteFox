
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = math.sqrt(self.head_num)
 
    def forward(self, query, key, value, dropout_p=0.2):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(batch_size, self.head_num, length_q, d_k)
key = torch.randn(batch_size, self.head_num, length_k, d_k)
value = torch.randn(batch_size, self.head_num, length_k, d_v)
dropout_p = 0.2
