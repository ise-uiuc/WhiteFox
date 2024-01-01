
class Model(torch.nn.Module):
    def __init__(self, query, key, value, inv_scale_factor, dropout_p: float = 0.25) -> None:
        super().__init__()
  
        self._dropout_p = dropout_p
        self._inv_scale_factor = inv_scale_factor
 
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        qk = torch.matmul(query, key.transpose(-2, -1)) 
        scaled_qk = qk.div(self._inv_scale_factor) 
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self._dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(query, key, value, inv_scale_factor)

# Inputs to the model
query = torch.randn(1, 256, 256)
key = torch.randn(1, 256, 256)
value = torch.randn(1, 256, 256)
