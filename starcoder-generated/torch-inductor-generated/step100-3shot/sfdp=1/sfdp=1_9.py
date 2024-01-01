
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout_p=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.scale_factor = torch.sqrt(torch.tensor(input_dim, dtype=torch.float))
        self.w_q = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.w_k = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.w_v = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x1, x2):
        q = self.w_q(x1) # Apply the query matrix to the input tensor containing queries
        k = self.w_k(x2) # Apply the key matrix to the input tensor containing keys
        v = self.w_v(x2) # Apply the value matrix to the input tensor containing values
        qk = torch.matmul(q, k.transpose(-2, -1)) / self.scale_factor # Compute the scaled dot product of the query and key tensors
        softmax_qk = qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        pooled_softmax_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = pooled_softmax_qk.matmul(v) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
m = Model(256, 512, 8)

# Inputs to the model
x1 = torch.randn(2, 8, 256)
x2 = torch.randn(2, 14, 256)
