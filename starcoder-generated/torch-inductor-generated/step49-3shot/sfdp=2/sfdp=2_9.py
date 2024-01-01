
class Model(torch.nn.Module):
    def forward(self, q_data, k_data):
        q = torch.randn(8, 8, 5, 1, device=q_data.device)
        k = torch.randn(8, 8, 5, 1, device=q_data.device)
        matmul1 = q.matmul(k.transpose(-2, -1))
        div1 = matmul1.div(torch.tensor(8.0, device=matmul1.device))
        softmax1 = div1.softmax(dim=-1)
        dropout = torch.nn.functional.dropout(softmax1, p=0.5)
        m = dropout.matmul(torch.randn(8, 8, 5, 10, device=dropout.device))
        return m

# Initializing the model
m = Model()

# Inputs to the model
q_data = torch.randn(1, 8, 5, 1, device="cpu")
k_data = torch.randn(1, 8, 5, 1, device="cpu")
