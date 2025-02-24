import wandb 
from tqdm import tqdm
import os
from dotenv import load_dotenv
import time

import torch
import torch.nn
import omegaconf
from omegaconf import OmegaConf
import logging

from utils import load_config#, set_logger
from model import EncoderForClassification
from data import get_dataloader
from transformers import set_seed
from accelerate import Accelerator

set_seed(42)

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)  # Get predicted class
    correct = (preds == label).sum().item()  # Count correct predictions
    total = label.size(0)  # Total predictions
    return correct, total

def main(configs : omegaconf.DictConfig, BSZ):
    start_time = time.time()
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = EncoderForClassification(configs).to(device)
    model.train()

    # Make Directories
    logging_dir = "exp_2/log_1"
    if os.path.exists(logging_dir) == False:
        os.makedirs(logging_dir)
    model_dir = "exp_2/model_1"
    if os.path.exists(model_dir) == False:
        os.makedirs(model_dir)

    # Load data
    train_dataloader = get_dataloader(configs, 'train')
    valid_dataloader = get_dataloader(configs, 'valid')
    test_dataloader = get_dataloader(configs, 'test')

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

    # Set accelerator
    accelerator = Accelerator(gradient_accumulation_steps=BSZ // configs.batch_size)
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )

    # Setting
    step = 0
    best_acc = 0
    best_acc_epoch = 0
    num_epoch = configs.epochs
    counter = tqdm(range(num_epoch*len(train_dataloader)), desc='Training :')
    ACCUM_STEP = BSZ / configs.batch_size
    print('ACCUM STEP:', ACCUM_STEP)

    # Set wandb
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(project='nlp_pretrain_exp2-1',
               name=f'[imdb] BSZ-{BSZ}',
               config=OmegaConf.to_container(configs, resolve=True, throw_on_missing=True))

    logging.basicConfig(
        filename=os.path.join(logging_dir, f'[imdb] BSZ-{BSZ}-train.log'),
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    logger = logging.getLogger()

    # Train & validation for each epoch
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                label = batch['label'].to(device)
                model_input = {key: value.to(device) for key, value in batch.items() if key != 'label'}
                outputs = model(**model_input, label=label)
                train_loss = outputs['loss']
                accelerator.backward(train_loss)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
            counter.update(1)
            if step % 100 == 0:
                wandb.log({"train_loss": train_loss.item()})
                print(f"step: {step}, train_loss: {train_loss:.4f}")
                logger.info(f"step: {step}, train_loss: {train_loss:.4f}")

        with torch.no_grad():
            model.eval()
            valid_loss = 0
            valid_step = 0
            val_correct = 0
            val_total = 0
            for batch in valid_dataloader:
                label = batch['label'].to(device)
                model_input = {key : value.to(device) for key, value in batch.items() if key != 'label'}
                outputs = model(**model_input, label=label)
                valid_loss += outputs['loss'].item()
                correct, total = calculate_accuracy(outputs['logits'], label)
                val_correct += correct
                val_total += total
                valid_step += 1
            epoch_val_loss = valid_loss/valid_step
            epoch_val_acc = val_correct / val_total
            print(f"valid_loss: {epoch_val_loss:.4f}")
            print(f"valid_acc: {epoch_val_acc:.4f}")
            # wandb logging
            wandb.log({"valid_loss": epoch_val_loss, "valid_acc": epoch_val_acc})
            logger.info(f"valid_loss: {epoch_val_loss:.4f}, valid_acc: {epoch_val_acc:.4f}")
            model.train()
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_acc_epoch = epoch+1

        torch.save(model.state_dict(), f'exp_2/model_1/BSZ-{BSZ}_{epoch+1}.pt')
        print('model saved!')

    # test after training
    with torch.no_grad():
        model.load_state_dict(torch.load(f'exp_2/model_1/BSZ-{BSZ}_{best_acc_epoch}.pt'))
        print('model loaded!')
        model.eval()
        test_loss = 0
        test_step = 0
        test_correct = 0
        test_total = 0
        for batch in tqdm(test_dataloader, desc='Testing :'):
            label = batch['label'].to(device)
            model_input = {key : value.to(device) for key, value in batch.items() if key != 'label'}
            outputs = model(**model_input, label=label)
            test_loss += outputs['loss'].item()
            correct, total = calculate_accuracy(outputs['logits'], label)
            test_correct += correct
            test_total += total
            test_step += 1
        avg_test_loss = test_loss/test_step
        test_acc = test_correct / test_total
        print(f"test_loss: {avg_test_loss:.4f}")
        print(f"test_acc: {test_acc:.4f}")
        wandb.log({"test_loss": avg_test_loss, "test_acc": test_acc})
        logger.info(f"test_loss: {avg_test_loss:.4f}, test_acc: {test_acc:.4f}")
    
    end_time = time.time()
    print(f"Total Time : {(end_time - start_time)/60} min")
    wandb.log({"total_time (min)": (end_time - start_time)/60})
    logger.info(f"Total Time : {(end_time - start_time)/60} min")

if __name__ == "__main__" :
    BSZ = 1024 # 64, 256, 1024
    configs = load_config()
    main(configs, BSZ)