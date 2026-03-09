import torch
from torch.utils.data import DataLoader 
import torch.nn as nn # loss functions and layers
from tqdm import tqdm # training progress bar
from model_transformer import TransformerRegressor
from preprocessing import train_dataset

# training parameters
SEQ_LEN = 24 # input window: 24 hours
BATCH_SIZE = 32 # mini batch size
EPOCHS = 3  # the number of repetitions of training 
LEARNING_RATE = 1e-3  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train dataloader (train with mini batch)
train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle= True)

# create model and send to the device
model = TransformerRegressor(seq_len= SEQ_LEN).to(device)

# loss function and optimizer
criterion = nn.MSELoss() # mean square error
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train() # training mode
    epoch_loss = 0 # loss for each epoch

    for batch_X, batch_y in tqdm(train_loader, desc=f"[Epoch {epoch + 1} / {EPOCHS}"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad() # reset the gradients from the previous step
        output = model(batch_X) # predict using model
        loss = criterion(output, batch_y) # calculate mse loss
        loss.backward() # calculate gradients
        optimizer.step() # update model parameters

        epoch_loss += loss.item()

    # print average loss at the end of each epoch
    avg_loss = epoch_loss / len(train_loader)

    print(f"Epoch [{epoch + 1} / {EPOCHS}] - Loss: [{avg_loss}]")

    # The end of the training save the model
    torch.save(model.state_dict(), "transformer_energy_model.pth")
    print("Model saved successfully.")