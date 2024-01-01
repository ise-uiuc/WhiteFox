
class Model(torch.nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.dim_model = dim_model
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        key = key.transpose(-2, -1)
        qk = torch.matmul(query, key)
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
    def init_weight(self):
        for weight in self.parameters():
            nn.init.trunc_normal_(weight.data, std=0.02)

# Initializing the model
dim_model = 32
m = Model(dim_model)

# Inputs to the model
query = torch.randn(2, 10, dim_model)
key = torch.randn(2, 15, dim_model)
value = torch.randn(2, 15, dim_model)
scale_factor = 0.5
dropout_p = 0.1
m(query, key, value, scale_factor, dropout_p)

