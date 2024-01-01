
class Model(torch.nn.Module):
    def __init__(self, dim, dropout_rate=0.8, dropout_p=0.8853064031187673):
        super().__init__()
 
        # Parameters
        self.dim = dim
 
        # Layers
        self.scale_factor = torch.nn.Parameter(torch.Tensor([dim ** -0.5]))
        self.dropout_rate = dropout_rate
        self.dropout_p = dropout_p
 
        self.q_matrix = torch.nn.Parameter(torch.Tensor(dim, dim))
        torch.nn.init.normal_(self.q_matrix)
        self.k_matrix = torch.nn.Parameter(torch.Tensor(dim, dim))
        torch.nn.init.normal_(self.k_matrix)
        self.v_matrix = torch.nn.Parameter(torch.Tensor(dim, dim))
        torch.nn.init.normal_(self.v_matrix)
        self.dropout = torch.nn.Dropout(dropout_rate)
 
    def forward(self, queries, keys, values):
        # Shape: (N, D, 1, 1)
        scale_factor = self.scale_factor.view(1, 1, 1)
        
        # Shape: (1, 1, K, D)
        q_matrix = self.q_matrix.t().view(1, 1, self.dim, self.dim)
        # Shape: (1, 1, D, K)
        k_matrix = self.k_matrix.t().view(1, 1, self.dim, self.dim)
        # Shape: (1, 1, K, D)
        v_matrix = self.v_matrix.t().view(1, 1, self.dim, self.dim)
        
        # Shape: (N, D, 1, K)
        qk = torch.matmul(queries, k_matrix).unsqueeze(2)
        # Shape: (N, D, 1, K)
        scaled_qk = (qk * scale_factor).softmax(-1)
        # Shape: (N, D, 1, K)
        dropout_qk = torch.nn.functional.dropout(scaled_qk, p=self.dropout_p)
        # Shape: (N, D, 1, K)
        output = torch.matmul(dropout_qk, v_matrix).squeeze(2)
        # Shape: (N, D)
        output = self.dropout(output)
        return output

# Initializing the model
dim = 128
m = Model(dim=dim)

# Inputs to the model
queries = torch.randn(1, dim)
keys = torch.randn(1, dim)
values = torch.randn(1, dim)
