
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def dot_product(self, query, key, scale_factor, dropout_p, dropout_train):
        # Compute the dot product of the query and key tensors
        if dropout_train:
            qk = torch.matmul(query, key.transpose(-2, -1))
        else:
            qk = torch.matmul(query, key.transpose(-2, -1).contiguous().dropout(p=dropout_p, train=dropout_train))
        
        # Scale the dot product by a factor
        scaled_qk = qk.mul(scale_factor)
        
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        if dropout_train:
            # Apply dropout to the softmax output
            dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        else:
            # Apply dropout to the softmax output
            dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p, train=dropout_train)
        
        # Compute the dot product of the dropout output and the value tensor
        output = torch.matmul(dropout_qk, value)
        
        return output
 
class Attention(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.scale_factor = torch.sqrt(torch.tensor(size=(1,), dtype=torch.float32, device="cuda:0", requires_grad=False).fill_(dropout_p))
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, dropout_train):
        output = self.dot_product(
            query,
            key,
            self.scale_factor,
            self.dropout_p,
            dropout_train
        )
        return output

# Initializing the model
m = Attention(dropout_p=0)
 
# Inputs to the model
query = torch.randn(1, 16, 32)
key = torch.randn(1, 16, 64)
value = torch.randn(1, 16, 64)
dropout_train = True
