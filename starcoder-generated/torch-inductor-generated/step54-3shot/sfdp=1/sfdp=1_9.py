
class Model(torch.nn.Module):
    def __init__(self,
                 weight: torch.Tensor):
        super().__init__()
        self.weight = weight
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                inv_scale_factor: torch.Tensor):

        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(m.weight)

# Inputs to the model
query = torch.randn(2, 4, 17, 17)
key = torch.randn(2, 12, 17, 17)
value = torch.randn(2, 12, 17, 17)
inv_scale_factor = torch.randn(1)
