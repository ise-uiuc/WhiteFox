
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Linear layers and softmax
        s_factor = 32
        d_p = 0.3
        self.query = torch.nn.Linear(256, 256)
        self.key = torch.nn.Linear(256, 256)
        self.key_t = torch.nn.Linear(256, 256)
        self.value = torch.nn.Linear(256, 256)
 
    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        k_t = self.key_t(x1)
        v = self.value(x1)
        # Compute dot product
        q_k = torch.matmul(q, k_t)
        # Scale the dot product
        q_k = q_k * s_factor
        # Apply softmax
        q_k = q_k.softmax(dim=-1)
        # Apply dropout
        q_k = torch.nn.functional.dropout(q_k, d_p)
        # Compute dot product
        out = torch.matmul(q_k, v)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
