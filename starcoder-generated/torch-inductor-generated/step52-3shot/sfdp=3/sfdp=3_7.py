

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 32
        self.num_heads = 4
        self.scale_factor = (self.dim // self.num_heads)**-0.5
        self.q_linear = torch.nn.Linear(self.dim, self.dim)
        self.k_linear = torch.nn.Linear(self.dim, self.dim)
        self.v_linear = torch.nn.Linear(self.dim, self.dim)
 
    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = self.scale_factor*qk
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1, training=True)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 32, 128)
