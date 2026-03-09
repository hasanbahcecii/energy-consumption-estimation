import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from model_transformer import TransformerRegressor
from preprocessing import test_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create model and load trained weights
model = TransformerRegressor(seq_len= 24).to(device)
model.load_state_dict(torch.load("transformer_energy_model.pth"))
model.eval() # evaluation mode

# load test data
test_loader = DataLoader(test_dataset, batch_size= 32)

# loss function
criterion = nn.MSELoss()

# keep predicted and real values in these lists
all_preds = []
all_targets = []
test_loss = 0

# predict in test set
with torch.no_grad(): # no need to gradient calculation
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        output = model(batch_X) # prediction
        loss = criterion(output, batch_y) # loss calculation for each epoch
        test_loss += loss.item()

        all_preds.extend(output.cpu().numpy()) # collect the predictions
        all_targets.extend(batch_y.cpu().numpy()) # collect the real values


# print average test loss
avg_test_loss = test_loss / len(test_loader) 
print(f"Test Loss: {avg_test_loss}")  

# visualization of predicted and real values
plt.figure()
plt.plot(all_targets, label = "Real Value", color = "blue")
plt.plot(all_preds, label = "Prediction", color= "red")
plt.title("Energy Consumption Prediction w Transformers")
plt.xlabel("Timestep")
plt.ylabel("Energy Consumption (Scaled Version)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()



