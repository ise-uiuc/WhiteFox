
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(8, 4)
        self.linear1 = torch.nn.Linear(4, 2)
        
    def forward(self, k, v, q):
        w = self.linear0(k)
        b = self.linear1(q) # Use the query to multiply the key
        
        w_b = w * b # Multiply the key and the query
        
        w_b_sum = torch.sum(w_b, dim=-1) # Sum the result along the feature dimension
        w_b_sum_exp = torch.exp(w_b_sum) # Apply exp to the sum value
        w_b_sum_exp_sum = torch.sum(w_b_sum_exp, dim=-1) # Use the sum dimension to sum
        w_b_sum_exp_sum_inv = 1 / w_b_sum_exp_sum # Inverse the sum of exp
        
        result = torch.sum(w * (b / w_b_sum_exp * w_b_sum_exp_sum_inv), dim=-1)
        
        return result
    
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 2, 8)
key = torch.randn(1, 5, 8)
value = torch.randn(1, 5, 8)
