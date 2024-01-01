
class Model(torch.nn.Module):

    def __init__(self, d_model, num_heads, dropout_p):
        super().__init__()
        self.d_model = d_model
        self.dropout_p = dropout_p
     
        self.queries_projection = torch.nn.Linear(d_model, d_model)
        self.keys_projection = torch.nn.Linear(d_model, d_model)
        self.values_projection = torch.nn.Linear(d_model, d_model)
        self.drop = torch.nn.Dropout(dropout_p)

    def forward(self, queries, keys, values):
        queries_projection = self.queries_projection(queries)
        keys_projection = self.keys_projection(keys)
        values_projection = self.values_projection(values)
 
        scaled_products = torch.matmul(queries_projection, keys_projection.transpose(-2, -1))
        scale_factor = (self.d_model ** (-0.5))
        scaled_products = scaled_products.mul(scale_factor)

        softmax_products = scaled_products.softmax(dim=-1)

        dropout_products = self.drop(softmax_products)
 
        product_output = torch.matmul(dropout_products, values_projection)
        return product_output

# Initializing the model
m = Model(d_model=128, num_heads=32, dropout_p=0.1)

# Inputs to the model
