
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.ones(1, 1))
 
    def forward(self, query, key, value, scale_factor=None, dropout_p=0.5):
        scale_factor = self.scale_factor if scale_factor is None else scale_factor
        # Reshape weight and input tensor
        shape = []
        for tensor in (query, key, value, scale_factor):
            new_shape = [1] * len(tensor.size())
            new_shape[0] = -1
            shape.append(new_shape)
        query = query.view(*shape)
        key = key.view(*shape)
        value = value.view(*shape)
        scale_factor = scale_factor.view(*shape[:-1])
        # Compute dot product
        qk = torch.matmul(query, key.transpose(-2, -1))
        # Scale
        scaled_qk = qk.mul(scale_factor)
        # Softmax
        shape2 = []
        for _ in range(len(tensor.size())+1):
            shape2.append(-1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        # Dropout
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        # Apply attention weights
        output = torch.matmul(dropout_qk.view(*shape2), value.view(*shape2))
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 64, 128)
key = torch.randn(1, 64, 256)
value = torch.randn(1, 64, 256)
