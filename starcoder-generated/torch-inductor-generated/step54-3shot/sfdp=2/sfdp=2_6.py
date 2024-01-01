
class Model(torch.nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout_p=0.1):
        super().__init__()
        # parameters
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout_p = dropout_p
        # layers
        self.query_proj = torch.nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.key_proj = torch.nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.value_proj = torch.nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1)) # Compute the dot product of the output of the query projection and the output of the key projection
        # Compute the inverse scale factor
        inv_scale_factor = np.power(self.d_head, -0.5)
        # Scale the dot product
        scaled_qk = qk.div(inv_scale_factor)
        # Apply softmax to the dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        # Apply dropout to the softmax output
        dropout_qk = self.dropout(softmax_qk)
        # Compute the dot product of the output of the value projection and the dropout output
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model(3, 32, 32, dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(1, 32, 1024)
x2 = torch.randn(1, 32, 2048)
