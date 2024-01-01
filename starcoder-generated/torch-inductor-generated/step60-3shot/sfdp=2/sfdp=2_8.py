
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 3
        self.out_features = 5
        self.drop_ratio = 0.3
        
        self.linear1 = torch.nn.Linear(self.in_features, 8)
        #...
        self.linear3 = torch.nn.Linear(32, self.out_features)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.linear2(x2)
        v3 = v1 + v2 # Element-wise add the output of the first layer and the output of the second layer
        v3 = v3.relu() # Apply relu to the output of the add
        v4 = self.linear3(v3) # Pass the output of the relu to the third layer
        v5 = v4.sigmoid() # Apply sigmoid to the output of the third layer
        dropout_v5 = torch.nn.functional.dropout(v5, p=self.drop_ratio) # Apply dropout to the output of the sigmoid layer
        return dropout_v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
