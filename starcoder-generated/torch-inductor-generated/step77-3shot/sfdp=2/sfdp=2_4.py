
class Model(torch.nn.Module):
    def __init__(self,
                 num_query,
                 num_key,
                 num_value,
                 dropout_p):
        super().__init__()
        self.num_query = num_query
        self.num_key = num_key
        self.num_value = num_value
        self.dropout_p = dropout_p
        self.qkv_projection = torch.nn.Linear(num_query + num_key + num_value, num_key)
        
        self.q_projection = torch.nn.Linear(num_query, num_value)
        self.k_projection = torch.nn.Linear(num_key + num_value, num_value)
        
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.output_projection = torch.nn.Linear(num_key, num_value)
        
    def forward(self, query, key, value):
        query = self.q_projection(query)
        key = self.k_projection(torch.cat([key, value], dim=-1))
        v1 = self.qkv_projection(torch.cat([query, key, value], dim=-1))
        v2 = v1 * (1. / (self.num_query**0.25)) # This is the inverse scale factor. The formula is (1 / sqrt(dim_1))
        v3 = self.softmax(v2)
        v4 = self.dropout(v3)
        output = self.output_projection(torch.matmul(v4, value))
        return output

# Initializing the model
m = Model(2, 6, 4, 0.5)

# Inputs to the model
query = torch.randn(2, 4)
key = torch.randn(2, 6)
value = torch.randn(2, 4)
