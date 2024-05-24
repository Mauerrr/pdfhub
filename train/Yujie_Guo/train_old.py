# modify based on Yujie_Guo version
import torch
from torch.utils.data import DataLoader
import os
import sys
import argparse
import datetime

current_file_path = os.path.abspath(__file__)
endtoenddriving_Path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))))
sys.path.append(endtoenddriving_Path)

from data_prep.Yujie_Guo.dataset_old import OpensceneDataset
import data_prep.Yujie_Guo.config_old as config
from network.Yujie_Guo.model_old import SimpleMLP, LSTMModel
from torch.utils.tensorboard.writer import SummaryWriter

def train_val(model, train_dataloader, val_dataloader, device, args, early_stopping):
    """
    Train and validate the model.

    Args:
        model (torch.nn.Module): The neural network model.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): Device for computation.
        args (argparse.Namespace): Parsed arguments.
    """
    print(f"Batch size is {args.batch_size}, one epoch has {len(train_dataloader)} iterations")
    main_tag = 'Loss'
    if args.overfit:
        main_tag = 'LossOverfit'
    # Declare the loss function and optimizer, using the value from CLI
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=config.betas, eps=config.eps,
                                 weight_decay=config.weight_decay)
    
    # Get the model name
    model_name = type(model).__name__
    
    # Get time to starting training
    start_train_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Get folder for tensorboard log
    tb_log_path = os.path.join(config.tb_log, f'{model_name}_tb_log_{start_train_time}')

    # TensorBoard writer
    writer = SummaryWriter(tb_log_path)

    # Set model to train, important if the network has dropout or batchnorm layers
    model.train()
    best_loss = float('inf')
    patience = args.patience_num  # Number of epochs to wait before early stopping
    wait = 0  # Initialize the waiting time

    for epoch in range(args.num_epochs):
        train_loss_running = 0.
        for i, batch in enumerate(train_dataloader):
            input_data, target_labels = batch
            # Move input_data and target_labels to device
            input_data, target_labels = input_data.to(device), target_labels.to(device)
            # 1 Zero out gradients from last iteration
            optimizer.zero_grad()
            # 2 Perform forward pass
            prediction = model(input_data)
            # 3 Calculate loss
            loss = criterion(prediction, target_labels)
            # 4 Compute gradients
            loss.backward()
            # 5 Adjust weights using the optimizer
            optimizer.step()

            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + i
            if iteration % args.log_interval == (args.log_interval - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / args.log_interval:.3f}')
                # Write train loss to TensorBoard
                writer.add_scalars(main_tag, {'train': train_loss_running / args.log_interval}, iteration)
                train_loss_running = 0.
            # Validation evaluation and logging
            if iteration % args.eval_freq == (args.eval_freq - 1):
                model.eval()
                loss_val = 0.
                for batch_val in val_dataloader:
                    input_data, target_labels = batch_val
                    input_data, target_labels = input_data.to(device), target_labels.to(device)
                    with torch.no_grad():
                        prediction = model(input_data)

                    loss_val += criterion(prediction, target_labels).item()

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_val / len(val_dataloader):.3f}')
                writer.add_scalars(main_tag, {'validation': loss_val / len(val_dataloader)}, iteration)

                # early stopping:
                if early_stopping.early_stop(loss_val / len(val_dataloader)):
                    # stop training phase
                    return
        
                # Saving the best checkpoints
                if loss_val / len(val_dataloader) < best_loss:
                    best_model_path = os.path.join(endtoenddriving_Path, 'train/Yujie_Guo/output')
                    # create best_model folder if not exists
                    if not os.path.exists(best_model_path):
                        os.makedirs(best_model_path)
                    torch.save(model.state_dict(), os.path.join(best_model_path, f'{model_name}_best_{start_train_time}.ckpt'))
                    best_loss = loss_val / len(val_dataloader)
                    # wait = 0  # Reset the waiting time
                # else:
                #     wait += 1

                # # Early stopping check    
                # if wait >= patience:
                #     print("Early stopping triggered!")
                #     return  # End training
                
                # Set model back to train
                model.train()
    writer.close()

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                return True
        return False

def parse_cfg():
    """
    Parse command line arguments for training.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run training with given parameters')

    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=config.num_epochs)
    parser.add_argument("--lr", type=int, help="starting learning rate for optimizer", default=config.lr)
    parser.add_argument("--eval_freq", type=int, help="validation frequency", default=config.eval_freq)
    parser.add_argument("--save_freq", type=int, help="save the model every save_freq steps", default=config.save_freq)
    parser.add_argument("--log_interval", type=int, help="logs output interval", default=config.log_interval)
    parser.add_argument("--device", type=str, help='Device to use for training', default=config.device,
                        choices=['cpu', 'cuda'])
    parser.add_argument("--overfit", type=bool, help='whether to run overfit', default=False)
    parser.add_argument("--batch_size", type=int, help="batch size", default=config.batch_size)
    parser.add_argument("--num_workers", type=int, help="number of workers", default=config.num_workers)
    parser.add_argument("--patience_num", type=int, help="patience number for early stopping", default=config.patience_num)
    return parser.parse_args()


def create_dataloader(dataset_name, batch_size, shuffle, num_workers):
    """
    Create a DataLoader for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset ('train', 'val', 'test', 'overfit').
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of DataLoader workers.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset.
    """
    dataset = OpensceneDataset(dataset_name)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=True)
    return dataloader


def main():
    """
    Main function for training the model.
    """
    args = parse_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print('Using device:', device)

    model = SimpleMLP().to(device)

    if args.overfit:
        train_dataloader = create_dataloader('overfit', config.batch_size, False, args.num_workers)
        val_dataloader = train_dataloader  # overfit dataset used for both training and validation
    else:
        train_dataloader = create_dataloader('train', args.batch_size, True, args.num_workers)
        val_dataloader = create_dataloader('val', args.batch_size, False, args.num_workers)

    early_stopper = EarlyStopper(patience=20, min_delta=0.0)

    train_val(model, train_dataloader, val_dataloader, device, args, early_stopper)


if __name__ == '__main__':
    main()

