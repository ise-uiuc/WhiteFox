
class Model(torch.nn.Module):
    def __init__(self,
                query, # A tensor for query
                key, # A tensor for key
                value, # A tensor for value
                inv_scale_factor, # 1.0 / sqrt(8)
                dropout_p, # 0.5
                ):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.matmul_1 = torch.matmul(query, key.transpose(-2, -1))
        self.matmul_2 = torch.matmul(self.softmax(self.matmul_1))
 
    def forward(self):
        return self.dropout(self.matmul_2)

# Initializing the model
m = Model(query, key, value, inv_scale_factor, dropout_p)

# Inputs to the model
