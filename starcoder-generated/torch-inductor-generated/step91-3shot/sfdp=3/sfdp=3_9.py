
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.softmax_qk = torch.nn.Softmax(dim=-1)
            self.dropout_qk = torch.nn.Dropout(dropout_p)
            self.output_proj = torch.nn.Linear(hidden_size, output_size)
 
        def forward(self, q, k, v, mask):
            qk = torch.matmul(q, k.transpose(-2, -1))
            scaled_qk = qk.mul(scale_factor)
            softmax_qk = self.softmax_qk(scaled_qk)
            dropout_qk = self.dropout_qk(softmax_qk)
            output = torch.matmul(dropout_qk, v)
            return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 16, 16, HIDDEN_SIZE)
k = torch.randn(1, 16, 16, HIDDEN_SIZE)
v = torch.randn(1, 16, 16, HIDDEN_SIZE)
mask = torch.ones(16, 16)
