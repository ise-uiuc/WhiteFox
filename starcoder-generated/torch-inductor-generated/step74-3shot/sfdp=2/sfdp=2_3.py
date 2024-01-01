
class Model(torch.nn.Module):
    def _init__(self, hidden_size, num_heads, dropout_p = 0.1, attn_dropout_p = 0.1):
        super(Model, self)._init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.attn_dropout_p = attn_dropout_p

        self.W_Q = torch.nn.Linear(hidden_size, num_heads * hidden_size) 
        self.W_K = torch.nn.Linear(hidden_size, num_heads * hidden_size) # The key transformation
        self.W_V = torch.nn.Linear(hidden_size, num_heads * hidden_size) # The value transformation
        self.w_O = torch.nn.Linear(num_heads * hidden_size, hidden_size)
 
    def forward(self, XQ, XK, XV):
        # Compute the dot product of the query and the key
        qk = torch.matmul(XQ, XK.transpose(-2, -1))
        # Scale the dot product by 1/sqrt(hidden_size)
        scaled_qk = qk.div(math.sqrt(self.hidden_size))
        # Apply softmax with the dot product as input
        softmax_qk = scaled_qk.softmax(dim = -1)
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p = self.dropout_p)
        # Compute the dot product of the dropout output and the value
        output = dropout_qk.matmul(XV)
        # Compute the weighted output
        weighted_output = self.w_O(output)
        return weighted_output

# Initializing the model
m = Model(hidden_size=32, num_heads=4)

# Inputs to the model (the batch size is 2, time steps is 4)
X_q = torch.randn(2, 4, 16)
X_k = torch.randn(2, 2, 16)
X_v = torch.randn(2, 2, 16)

