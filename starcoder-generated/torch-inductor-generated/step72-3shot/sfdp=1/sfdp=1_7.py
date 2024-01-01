
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, mask):
        qk = torch.matmul(q, k.transpose(-2, -1))
        s1 = qk.shape[0]
        s2 = qk.shape[1]
        qk = torch.reshape(qk, (1,s1,s2,qk.shape[2],qk.shape[3]))
        s3 = qk.shape[4]
        mask = torch.reshape(mask, (s1,s2,1,1,s3))
        qk = qk.masked_fill(mask == 0, -1e3)
        inv_scale_factor = 1. / math.sqrt(s3)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.0)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Dimensions of query, key, value and mask
q_dim = 20
k_dim = 30
v_dim = 40
h_dim = 50
batch = 10
query = torch.randn(batch, q_dim, h_dim)
key = torch.randn(batch, k_dim, h_dim)
value = torch.randn(batch, v_dim, h_dim)
mask = torch.zeros((batch, k_dim, 1, 1))
