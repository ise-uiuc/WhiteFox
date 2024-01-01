
class Scaled_Dot_Attention(torch.nn.Module):
    def __init__(self, d_model, q_dim, k_dim, dropout_p):
        super().__init__()
        self.q = torch.nn.Linear(q_dim, k_dim, bias=False)
        self.k = torch.nn.Linear(k_dim, k_dim, bias=False)
        self.v = torch.nn.Linear(d_model, k_dim, bias=False)
        self.dr_k = torch.nn.Dropout(dropout_p)
        self.dr_v = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=1)
        self.inv_scale_factor = torch.nn.Parameter(
            torch.tensor(1.0 / (k_dim ** 0.5))
        )
 
    def forward(self, queries, keys, values):
        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)
        q_k = torch.matmul(q, k.transpose(-2, -1))
        scaled_q_k = q_k.div(self.inv_scale_factor)
        dr_scaled_q_k = self.dr_k(scaled_q_k)
        softmax_scaled_q_k = self.softmax(dr_scaled_q_k)
        dropout_softmax_scaled_q_k = self.dr_v(softmax_scaled_q_k)
        output = torch.matmul(dropout_softmax_scaled_q_k, v)
        return output

class Model(torch.nn.Module):
    def __init__(self, d_model, q_dim, k_dim, dropout_p):
        super().__init__()
        self.scaled_dot_attention = Scaled_Dot_Attention(d_model, q_dim, k_dim, dropout_p)
 
    def forward(self, x1, x2):
        output = self.scaled_dot_attention(x1, x2, x2)
        return output

# Initializing the model
d_model = 8
q_dim = 8
k_dim = 8
dropout_p = 0.01
m = Model(d_model, q_dim, k_dim, dropout_p)

# Inputs to the model
x1 = torch.randn(1, 3, 10)
x2 = torch.randn(1, 3, 18)
