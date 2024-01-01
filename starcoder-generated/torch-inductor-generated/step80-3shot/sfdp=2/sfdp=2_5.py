
class Model(torch.nn.Module):
    def forward(self, query, key, value, scale_factor, dropout_p):
        s1 = torch.matmul(query, key.transpose(-2, -1))
        s2 = s1 / scale_factor
        s3 = torch.nn.functional.softmax(s2, dim=-1)
        o1 = torch.nn.functional.dropout(s3, p=dropout_p)
        o2 = torch.matmul(o1, value)
        return o2

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn((1, 16, 5))
key = torch.randn((1, 16, 10))
value = torch.randn((1, 16, 10))
scale_factor = 1/math.sqrt(512)
dropout_p = 0.2
