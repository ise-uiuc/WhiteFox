
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_mask, dropout_p, dropout_mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div((scale_mask+1e-5).sqrt()) # Compute the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, dropout_p, training=True)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
scale_mask = torch.randn(1, 4, 8)
query = torch.randn(1, 2, 4, 8)
key = torch.randn(1, 2, 8, 8)
value = torch.randn(1, 2, 4, 8)
dropout_p = 0.1
dropout_mask = torch.randn(1, 4, 4)
