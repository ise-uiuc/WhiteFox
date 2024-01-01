
class Model(torch.nn.Module):
    __init__(self, dim_seq, num_heads, scale_factor):
        super().__init__()
        self.dim_seq = dim_seq
        self.num_heads = num_heads
        self.scale_factor = scale_factor
 
    def forward(self, query, key, value, dropout_p):
        result = torch.matmul(query, key.transpose(-2, -1)) * self.scale_factor
        result = result.softmax(dim=-1)
        result = torch.nn.functional.dropout(result, p=dropout_p)
        result = result.matmul(value)
        return result
 
    @staticmethod
    def get_inputs(query, key, value, dropout_p):
        shape = list(query.shape)
        if len(shape) == 2:
            shape.insert(1, 1)
        elif len(shape) < 2 or len(shape) > 4:
            raise Exception
        dim_seq = shape[-1]
        num_heads = shape[-2]
        if shape!= key.shape or shape!= value.shape:
            raise Exception
        scale_factor = 1 / np.sqrt(dim_seq)
        return query, key, value, dim_seq, num_heads, scale_factor
 
m = Model(2, 2, 2)
query = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
key = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
value = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
dropout_p =.05 
