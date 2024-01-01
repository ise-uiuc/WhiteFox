
class Model(torch.nn.Module):
    def __init__(self, head_dim, num_heads, dropout_p=0.1, scale_factor=None):
        super(Model, self).__init__()
        self.dropout_p = dropout_p
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale_factor = scale_factor
        self.w_query = nn.Linear(self.head_dim, self.head_dim)
        self.w_key = nn.Linear(self.head_dim, self.head_dim)
        self.w_value= nn.Linear(self.head_dim, self.head_dim)
 
    def forward(self, x1, x2):
        query = self.w_query(x1)
        key = self.w_key(x1)
        value = self.w_value(x1)
        
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))

        # Scale the dot product by the inverse scale factor
        if self.scale_factor is not None:
            inv_scale_factor = 1 / self.scale_factor
            scaled_qk = qk.div(inv_scale_factor)
        else:
            scaled_qk = qk

        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)

        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)

        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(value)
        return output
# Initializing the model
m = Model(10, 2)

# Inputs to the model
x1 = torch.randn(3, 10)
x2 = torch.randn(1, 10)
