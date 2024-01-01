
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model()
 
# Inputs to the model
query = __torch__.rand([16, 2, 16])
key = __torch__.rand([16, 2, 32])
value = __torch__.rand([16, 2, 32])
# Only support scale_factor = 1/sqrt(embedding_dim)
scale_factor = torch.sqrt(torch.tensor(key.size(-1)).to(key.device))
dropout_p = 0.05
