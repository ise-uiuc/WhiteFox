
class Model(torch.nn.Module):
    def __init__(self, *, d_model: int, dropout_p: float):
        super().__init__()
        scale_factor = torch.sqrt(torch.FloatTensor([d_model]))
        self.scale_factor = scale_factor / scale_factor.numel()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        scale_factor = self.scale_factor.to(query.device)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
d_model = 32
dropout_p = 0.5
m = Model(d_model=d_model, dropout_p=dropout_p)

# Inputs to the model
batch_size = 64
num_head = 4
head_dim = d_model // num_head
query = torch.randn(batch_size, num_head, head_dim, head_dim)
key = torch.randn(batch_size, num_head, head_dim, head_dim)
value = torch.randn(batch_size, num_head, head_dim, head_dim)
