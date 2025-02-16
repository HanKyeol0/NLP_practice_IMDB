import wandb 
from tqdm import tqdm
import os
from dotenv import load_dotenv

import torch
import torch.nn
import omegaconf
from omegaconf import OmegaConf
import logging

from utils import load_config#, set_logger
from model import EncoderForClassification
from data import get_dataloader
from transformers import set_seed

set_seed(42)

# torch.cuda.set_per_process_memory_fraction(11/24) -> 김재희 로컬과 신입생 로컬의 vram 맞추기 용도. 과제 수행 시 삭제하셔도 됩니다. 
# model과 data에서 정의된 custom class 및 function을 import합니다.
"""
여기서 import 하시면 됩니다. 
"""

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)  # Get predicted class
    correct = (preds == label).sum().item()  # Count correct predictions
    total = label.size(0)  # Total predictions
    return correct, total

def main(configs : omegaconf.DictConfig, model_name):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = EncoderForClassification(configs).to(device)
    model.train()

    # Load data
    train_dataloader = get_dataloader(configs, 'train')
    valid_dataloader = get_dataloader(configs, 'valid')
    test_dataloader = get_dataloader(configs, 'test')

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

    # Setting
    step = 0
    best_acc = 0
    best_acc_epoch = 0
    num_epoch = configs.epochs
    counter = tqdm(range(num_epoch*len(train_dataloader)), desc='Training :')

    # Set wandb
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(project='nlp_pretrain_exp1',
               name=f'[imdb] {model_name}',
               config=OmegaConf.to_container(configs, resolve=True, throw_on_missing=True))
    logging_dir = "log"
    if os.path.exists(logging_dir) == False:
        os.makedirs(logging_dir)
    logging.basicConfig(
        filename=os.path.join(logging_dir, f'[imdb] {model_name}-train.log'),
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    logger = logging.getLogger()

    # Train & validation for each epoch
    for epoch in range(num_epoch):
        for batch in train_dataloader:
            label = batch['label'].to(device)
            model_input = {key : value.to(device) for key, value in batch.items() if key != 'label'}
            optimizer.zero_grad()
            outputs = model(**model_input, label=label)
            train_loss = outputs['loss']
            train_loss.backward()
            optimizer.step()
            step += 1
            counter.update(1)
            wandb.log({"train_loss": train_loss.item()})
            if step % 100 == 0:
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

        torch.save(model.state_dict(), f'exp_1/model/{model_name}_{epoch+1}.pt')
        print('model saved!')

    # test after training
    with torch.no_grad():
        model.load_state_dict(torch.load(f'exp_1/model/{model_name}_{best_acc_epoch}.pt'))
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
    
if __name__ == "__main__" :
    model_name='bert-base-uncased'#'ModernBERT-base' # or 'bert-base-uncased'
    configs = load_config(model=model_name)
    main(configs, model_name)