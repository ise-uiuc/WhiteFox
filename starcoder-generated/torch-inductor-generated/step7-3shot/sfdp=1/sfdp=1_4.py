
class SelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.matmul = torch.nn.MatMul()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(config.hidden_dropout)
        self.matmul1 = torch.nn.MatMul()
 
    def forward(self, query, key, value, mask, inv_scale_factor):
        qk = self.matmul(query, key)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = self.matmul1(dropout_qk, value)
        if mask is not None:
            output = output.masked_fill(mask.to(torch.bool), -10000)
        return output

# Initializing the model
model = SelfAttention(default_transformer_config())

# Inputs to the model
query = torch.randn(2, 8, 128)
key = torch.randn(2, 8, 128)
value = torch.randn(2, 8, 128)
mask = torch.tensor([[0, 1, 0], [1, 0, 0]]).unsqueeze(1)
inv_scale_factor = torch.tensor([0.5])
