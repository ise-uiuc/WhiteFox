
class Model(torch.nn.Module):
    def _init__(self, query_size, key_size, value_size, dropout_p):
        super().__init__()
 
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        inv_scale_factor = torch.tensor(hidden_size ** -0.5)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Inputs to the model
query_size = 512
key_size = 512
value_size = 512
dropout_p = 0.0
query = torch.randn(1, 100, query_size)
key = torch.randn(1, 1000, key_size)
value = torch.randn(1, 1000, value_size)
___output___ = Module(query_size, key_size, value_size, dropout_p)(query, key, value)

