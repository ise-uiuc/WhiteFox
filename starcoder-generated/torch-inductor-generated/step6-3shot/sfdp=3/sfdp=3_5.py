
class Model(torch.nn.Module):
    def __init__(self, query_tensor, key_tensor, value_tensor):
        super().__init__()
        self.query_tensor = query_tensor
        self.key_tensor = key_tensor
        self.value_tensor = value_tensor
        self.scale_factor = self.query_tensor.shape[-1]**-.5
        self.dropout_p = 0.2
 
    def forward(self):
        v1 = torch.matmul(self.query_tensor, self.key_tensor.transpose(-2, -1))
        v2 = v1 * self.scale_factor
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, self.dropout_p)
        v5 = torch.matmul(v4, self.value_tensor)
        return v5

# Initializing the model
m = Model(query_tensor, key_tensor, value_tensor)

# Inputs to the model
