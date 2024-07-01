import os
import torch.distributed as dist
from tqdm import tqdm
from utils.dataloaders import main_dataset
from utils.models import davit, ComparativeModel
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import torch.nn as nn
import pandas as pd
import json

def train(train_df, val_df,MODEL_OUTPUTS,index):
    # parameters
    EPOCHS       = 1
    LEARNING_RATE_1 = 1e-3
    LEARNING_RATE_2 = 1e-3
    # OUTPUT_SHAPE = 2
    IMAGE_SIZE = 224
    BATCH_SIZE = 48
    EARLY_STOP_PATIENCE = 10
    
    # output folder creation
    if not os.path.exists(MODEL_OUTPUTS):
        os.mkdir(MODEL_OUTPUTS)
    
    # torch.manual_seed(0)

    # model initiation
    device = torch.device(f"cuda:{index}")
    image_encoder = davit().to(device)
    model = ComparativeModel(image_encoder, IMAGE_SIZE, device)
    model = model.to(device)

    
    tea_encoder_parameters = image_encoder.parameters()
    new_parameters = [param for param in model.parameters() if param not in tea_encoder_parameters]

    # Set different learning rates for each parameter group
    parameters = [
        {'params': tea_encoder_parameters},
        {'params': new_parameters, 'lr': LEARNING_RATE_2},
    ]

    optimizer = Adam(parameters, lr=LEARNING_RATE_1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    #dataloaders
    train_loader, val_loader = main_dataset(train_df, val_df, BATCH_SIZE)
    
    
    # load the model
    weight_dict = torch.load('/home/tishan/TeaRetina_AI_Experiments/checkpoints/Checkpoint_davit_smallNN_14_5_2024/Model-epoch79.pth')
    model.load_state_dict(weight_dict)
    
    
    #initiate
    best_val_loss = float('inf')
    patience_counter = 0

    # training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', dynamic_ncols=True) as pbar: # show the progress bar
            
            correct_count = 0.0
            for i, data in enumerate(pbar, 0):
                image1, image2, act_label = data

                image1, image2, act_label = image1.to(device), image2.to(device), act_label[0]
                act_label = act_label.to(device)

                optimizer.zero_grad()
                
                pred_label = model(image1, image2)
                
                # print(f'actual label : {act_label}')
                # print(f'predicted label : {pred_label}')
                
                loss = criterion(pred_label, act_label)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                correct_count += torch.sum(torch.argmax(pred_label, dim=1) == torch.argmax(act_label, dim=1))

                pbar.set_postfix({'Running Loss': running_loss / (i + 1)})

            train_acc = correct_count / len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            n_correct = 0.0

            with torch.no_grad():
                for i, data in enumerate(val_loader, 0):
                    inputs1, inputs2, val_act_label = data
                    inputs1, inputs2, val_act_label = inputs1.to(device), inputs2.to(device), val_act_label[0]

                    #val_act_label = torch.Tensor(val_act_label)
                    val_act_label = val_act_label.to(device)

                    val_pred_label = model(inputs1, inputs2)

                    loss = criterion(val_pred_label, val_act_label)

                    val_loss += loss.item()
                    
                    n_correct += torch.sum(torch.argmax(val_pred_label, dim=1) == torch.argmax(val_act_label, dim=1))
                    
            # Logging
            avg_train_loss = running_loss / len(train_loader.dataset)
            avg_val_loss = val_loss / len(val_loader.dataset)


            # Accuracies
            avg_acc = n_correct / len(val_loader.dataset)

            # Print validation scores after each epoch
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%, Val Accuracy: {avg_acc*100:.2f}%")

            # Early Stopping and Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(MODEL_OUTPUTS, f'Model-epoch{epoch+1}.pth'))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print("Early stopping")
                    break
            
            # scheduler    
            scheduler.step(avg_val_loss)
            
            #save to .json file
            loss_dict = {'train loss': [], 'val loss' : []}
            
            json_file_path = '/home/tishan/TeaRetina_AI_Experiments/loss/loss_dict_davit_smallNN_15_5_2024.json'
            
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as json_file:
                    loss_dict = json.load(json_file)
            
            loss_dict['train loss'].append(avg_train_loss)
            loss_dict['val loss'].append(avg_val_loss)
            
            with open(json_file_path, "w") as json_file:
                json.dump(loss_dict, json_file, indent=4)
                                

if __name__ == "__main__":  
    train_df = pd.read_csv('/home/tishan/TeaRetina_AI_Experiments/datasets/Amila Answers/new_dataset_15_5_2024/Dataset_1_train.csv')
    val_df = pd.read_csv('/home/tishan/TeaRetina_AI_Experiments/datasets/Amila Answers/new_dataset_15_5_2024/Dataset_1_val.csv')
    model_output_path = '/home/tishan/TeaRetina_AI_Experiments/checkpoints/Checkpoint_davit_smallNN_15_5_2024'
    train(train_df, val_df,model_output_path,2)
    
