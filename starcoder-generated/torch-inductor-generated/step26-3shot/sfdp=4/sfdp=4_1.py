
class Model(torch.nn.Module):
    def attention_mask(self, q):
        dim = q.shape[1] # Get the size of the batch dimension
        mask = torch.tril(torch.ones(dim, dim)) # Create a matrix of size dim by dim, with elements in the lower triangle being 1, and the upper triangle being 0
        if self.device == "cpu":
            mask = mask.unsqueeze(0) # Unsqueeze the matrix to add the batch dimension to the mask
        else:
            mask = mask.unsqueeze(0).cuda() # Unsqueeze the matrix to add the batch dimension to the mask, and move the mask to GPU
        return mask
 
    def forward(self, x1, x2):
        qk = x1 @ x2.transpose(-2, -1) / math.sqrt(x1.size(-1)) # Compute the dot product between the query and key, and scale it
        qk = qk + self.attention_mask(x1) # Compute the attention weights
        attn_weight = torch.softmax(qk, dim=-1) # Apply softmax on the attention weights
        output = attn_weight @ x2 # Compute the output by computing the dot product of the attention weights and the value
        return output

# Initializing the model
m = Model()
if torch.cuda.is_available():
    m = m.cuda()
    x1 = x1.cuda()
    x2 = x2.cuda()
with torch.no_grad():
    