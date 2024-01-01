
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(64, 64)
        self.query = torch.nn.Linear(64, 64)
    
    def forward(self, v1):
        k = self.key(v1)
        q = self.query(v1)
        mat = torch.matmul(q, k.transpose(-1, -2))
        scaled_mat = mat / 140737488355328
        softmax_mat = torch.nn.functional.softmax(scaled_mat, dim=-1)
        dropout_mat = torch.nn.functional.dropout(softmax_mat, p=0.1)
        output = torch.matmul(dropout_mat, v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 32, 64)
