
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    # A model with positional arguments
    def forward(self, hidden_state, attn_mask=1):
        h_r = hidden_state
        h_r = h_r + attn_mask   # Apply the attention mask
        attn_weight = torch.softmax(h_r, dim=-1)
        output = attn_weight @ hidden_state
        return output
# Inputs to the model
Q = torch.randn(1, 32, 33)
V = torch.randn(1, 32, 33)
mask = torch.rand(1, 33)
# Inputs ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 32)
    # A model without positional arguments
    def forward(self, hidden_state):
        h_r = self.linear1(hidden_state)
        h_r = h_r + h_r
        attn_weight = torch.softmax(h_r, dim=-1)
        output = attn_weight @ hidden_state
        return output
# Inputs to the model
Q = torch.randn(1, 32, 16)
V = torch.randn(1, 32, 16)
mask = torch.rand(1, 33)
# Inputs ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask):
        Q9 = Q.transpose(-2, -1)
        K0 = K.transpose(-2, -1)
        v9 = V.transpose(-2, -1)
        kq9 = K0 @ v9 / math.sqrt(K0.size(-1))
        kq9 = kq9 + mask
        weights = torch.softmax(kq9, dim=-1)
        qv = Q @ V / math.sqrt(Q.size(-1))
        qv = qv + masks
        output = weights @ qv
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
# Inputs ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Q9 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.K0 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.v9 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
    def forward(self):
        kq9 = self.K0 @ self.v9 / math.sqrt(self.K0.size(-1))
        weights = torch.softmax(kq9 + self.mask, dim=-1)
        qv = self.Q9 @ self.V / math.sqrt(self.Q9.size(-1)) + self.mask
        output = weights @ qv
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
# Inputs ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Q9 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.K0 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.v9 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
    def forward(self):
        kq9 = self.K0 @ self.v9 / math.sqrt(self.K0.size(-1))
        weights = torch.softmax(kq9 + self.mask, dim=-1)
        qv = self.Q9 @ V / math.sqrt(self.Q9.size(-1)) + mask
        output = weights @ qv
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
# Inputs ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Q9 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.K0 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.v9 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
    def forward(self):
        kq9 = self.K0 @ self.v9 / math.sqrt(self.K0.size(-1))
        weights = torch.softmax(kq9 + self.mask, dim=-1)
        qv = self.Q9 @ self.v9 / math.sqrt(self.Q9.size(-1)) + self.mask
        output = weights @ qv
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
# Inputs ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Q9 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.K0 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.v9 = torch.randn(1, 64, 56, 56).transpose(-2, -1)
        self.mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
    def forward(self):
        kq9 = self.K0 @ self.v9 / math.sqrt(self.K0.size(-1))
        weights = torch.softmax(kq9 + self.mask, dim=-1)
        qv = self.Q9 @ self.v9 / math.sqrt(self.Q9.size(-1)) + self.mask
        outputs = weights @ qv
        return outputs
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
# Inputs ends


# Please enter your choice for the model type

model_type = "custom"  # Enter correct option

if not model_type in ("custom", "pytorch"): raise Exception("Select valid model type: pytorch or custom.")

# Model starts
model_class = {
    "pytorch": Model()
}[model_type]  # Choose based on specified model type
