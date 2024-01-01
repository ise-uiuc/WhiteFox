
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(8, 1, bias=True)
        self.v_proj = torch.nn.Linear(8, 1, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, input):
        _qkv_out = self.qkv_proj(input)
        qkv_out = _qkv_out.chunk(len(_qkv_out.shape) >> 1, dim=-1)
        query = qkv_out[0]
        key = qkv_out[1]
        value = qkv_out[2]
        scale_factor = value.size(self.v_proj.weight.shape[0]) ** -0.5
        _scaled_qk = torch.matmul(query, key.transpose(-2, -1) * scale_factor)
        scaled_qk = _scaled_qk.div(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        _dropout_qk = self.dropout(softmax_qk)
        dropout_qk = _dropout_qk.matmul(value)
        output = dropout_qk.squeeze()
        return output

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(2, 8)
