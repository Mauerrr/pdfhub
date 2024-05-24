import torch
from torch.utils.data import DataLoader
import os
import sys
import argparse
from datetime import datetime

current_file_path = os.path.abspath(__file__)
endtoenddriving_Path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))))
sys.path.append(endtoenddriving_Path)

from data_prep.Yujie_Guo.dataset import NuplanDataset
import data_prep.Yujie_Guo.config as config
from network.Yujie_Guo.model import UrbanDriverModel
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
        early_stopping : Early stopping criterion
    """
    print(f"Batch size is {args.batch_size}, one epoch has {len(train_dataloader)} iterations")
    main_tag = 'Loss'
    writer_tag = 'TrainVal'
    if args.overfit:
        main_tag = 'LossOverfit'
        writer_tag = 'Overfit'
    # Declare the loss function and optimizer, using the value from CLI
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=config.betas, eps=config.eps,
                                 weight_decay=config.weight_decay)
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(config.tb_log, f"runs/{writer_tag}_{config.ego_state_model_part}_{config.preprocess_choice}_{datetime.now().strftime('%m_%d_%H_%M')}"))

    # Set model to train, important if the network has dropout or batchnorm layers
    model.train()
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        train_loss_running = 0.
        for i, batch in enumerate(train_dataloader):
            input_data, target_labels = batch
            # Move input_data and target_labels to device
            input_data, target_labels = input_data.to(device), target_labels.to(device)
            
            # Used for processing target
            delta_sum = torch.zeros_like(target_labels).to(device) # [batch_size, 8, 2]
            
            # 1 Zero out gradients from last iteration
            optimizer.zero_grad()
            # 2 Perform forward pass
            if config.ego_state_model_part == 'transformer':
                prediction = model(input_data, target_labels)
            else:
                prediction = model(input_data)
            
            # choose processing method for targets
            if config.preprocess_choice == 'subtract_adjacent_row_using_outputs':
                # print('Process for targets: subtract_adjacent_row_using_outputs'
                delta_sum[:, 1:, :] = torch.cumsum(prediction[:, :-1, :], dim=1)
                # process targets
                target_labels_processed = target_labels - delta_sum

            else:
                # print('Process for targets: subtract_first_row')
                target_labels_processed = target_labels
                
            # 3 Calculate loss
            loss = criterion(prediction, target_labels_processed)
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

                    # Used for processing target
                    delta_sum = torch.zeros_like(target_labels).to(device) # [batch_size, 8, 2]

                    with torch.no_grad():
                        if config.ego_state_model_part == 'transformer':
                            prediction = model(input_data, target_labels)
                        else:
                            prediction = model(input_data)

                    # choose processing method for targets
                    if config.preprocess_choice == 'subtract_adjacent_row_using_outputs':
                    # print('Process for targets: subtract_adjacent_row_using_outputs')
                        delta_sum[:, 1:, :] = torch.cumsum(prediction[:, :-1, :], dim=1)
                    # process targets
                        target_labels_processed = target_labels
                        predicetion_processed = torch.cumsum(prediction, dim=1)

                    else:
                        # print('Process for targets: subtract_first_row')
                        target_labels_processed = target_labels
                        predicetion_processed = prediction

                    loss_val += criterion(predicetion_processed, target_labels_processed).item()

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_val / len(val_dataloader):.3f}')
                writer.add_scalars(main_tag, {'validation': loss_val / len(val_dataloader)}, iteration)
                # early stopping:
                if early_stopping.early_stop(loss_val / len(val_dataloader)):
                    # stop training phase
                    return
                # Saving the best checkpoints
                if loss_val / len(val_dataloader) < best_loss:
                    best_model_path = os.path.join(endtoenddriving_Path, 'train/Yujie_Guo/best_model')
                    # create best_model folder if not exists
                    if not os.path.exists(best_model_path):
                        os.makedirs(best_model_path)
                    torch.save(model.state_dict(), os.path.join(best_model_path, f'model_best_{config.ego_state_model_part}.ckpt'))
                    best_loss = loss_val / len(val_dataloader)
                # Set model back to train
                model.train()
    writer.close()


class EarlyStopper:
    def __init__(self, patience=30, min_delta=0.):
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
    dataset = NuplanDataset(dataset_name)
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

    model = UrbanDriverModel(config.ego_state_model_part).to(device)

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

