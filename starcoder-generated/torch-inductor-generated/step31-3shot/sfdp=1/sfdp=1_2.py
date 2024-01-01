
class Model(torch.nn.Module):
    def __init__(self, batch_size, dim_q, dim_k, dim_v, num_heads, dropout_p):
        super().__init__()
        # Parameters
        self.dropout_p = dropout_p
 
        # Query, key, and value linear transformations
        self.q = torch.nn.Linear(dim_q, dim_q)
        self.k = torch.nn.Linear(dim_k, dim_k)
        self.v = torch.nn.Linear(dim_v, dim_v)
 
        # Dropout
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, x1):
        x2 = torch.nn.functional.relu(self.q(x1))
        x3 = torch.nn.functional.relu(self.k(x1))
        x4 = torch.nn.functional.relu(self.v(x1))
 
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(x2, x3.transpose(-2, -1))
 
        # Scale the dot product by the inverse scale factor
        scaled_qk = qk.div(np.sqrt(self.dim_k))
 
        # Apply softmax to the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
 
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
 
        # Compute the dot product of the dropout output and the value tensor
        output = dropout_qk.matmul(x4)
 
        return output
 
