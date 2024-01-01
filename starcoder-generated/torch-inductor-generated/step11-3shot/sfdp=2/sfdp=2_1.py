
class Model(torch.nn.Module):
    def __init__(self, input_dim, num_heads, head_dim, dropout_p=0.0):
        super().__init__()
        
        # The number of input dimensions of the query, key, and value
        self.input_dim = input_dim
        
        