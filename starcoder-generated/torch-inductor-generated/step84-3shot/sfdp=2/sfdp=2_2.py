
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, dropout_p=0):
        return torch.nn.functional.dropout(
            torch.nn.functional.softmax(
                torch.matmul(query, key.transpose(-2, -1)).div(math.sqrt(512))
            ),
            p=dropout_p).matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.rand(1024, 512)
key = torch.rand(1024, 512)
value = torch.rand(1024, 512)
