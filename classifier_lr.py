import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
import torch.nn as nn
# change it with respect to the original model
from tokenizer import BertTokenizer
from bert import BertModel
# from optimizer import AdamW
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
TQDM_DISABLE=False
# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class BertSentimentClassifier(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''
    def __init__(self, config):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('./bert-base-uncased')
        # Pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        ### TODO

        # linear layer
        self.classify = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size,self.num_labels),
        )


    def forward(self, input_ids, attention_mask):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: you should consider what is the appropriate output to return given that
        # the training loop currently uses F.cross_entropy as the loss function.
        ### TODO
        # get BERT embedding for batch of sentences
        outputs = self.bert(input_ids, attention_mask)
        sentiment_logits=self.classify(outputs['pooler_output'])

        # pooler_output = outputs['pooler_output']
        # pooler_output = self.dropout(pooler_output)
        # sentiment_logits = self.linear(pooler_output)

        return sentiment_logits



class SentimentDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data

class SentimentTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data

# Load the data: a list of (sentence, label)
def load_data(filename, flag='train'):
    num_labels = {}
    data = []
    if flag == 'test':
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                data.append((sent,sent_id))
    else:
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label,sent_id))
        print(f"load {len(data)} data from {filename}")

    if flag == 'train':
        return data, len(num_labels)
    else:
        return data

# Evaluate the model for accuracy.
def model_eval(dataloader, model, device):
    model.eval() # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                        batch['labels'], batch['sents'], batch['sent_ids']
                                                      

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids


def model_test_eval(dataloader, model, device):
    model.eval() # switch to eval model, will turn off randomness like dropout
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                         batch['sents'], batch['sent_ids']
                                                      

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = BertSentimentClassifier(config)
    model = model.to(device)

    lr = args.lr
    parameters=get_parameters(model,5e-5,0.95, 3e-5)

## 4e-5,0.95, 2e-5 0.529

## 5e-5,0.95, 5e-5 0.529
## 4e-5,0.95, 2e-5 0.537
## 5e-5,0.95, 2e-5 0.532
## 5e-5,0.9,  1e-4 0.518
## 4e-5,0.95, 5e-5 0.528



## 3e-5,0.9,  3e-5 0.515
## 4e-5,0.9,  2e-5 0.527
## 5e-5,0.95, 2e-5 0.534
## 5e-5,0.9,  2e-5  0.524
## 4e-5,0.95, 1e-5 0.520
## 4e-5,0.95, 2e-5 0.539
## 4e-5,0.95, 3e-5 0.520
## 4e-5,0.95, 4e-5 0.520
## 4e-5,0.95, 5e-5 0.533
## 5e-5,0.95, 5e-5 0.535
## 5e-5,0.9, 5e-5 0.530
## 5e-5,0.9, 2e-5 0.530

    optimizer=AdamW(parameters)
    # optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    train_accuracies=[]
    val_accuracies=[]
    train_losses=[]

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, *_  = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        train_accuracies.append(train_acc)
        val_accuracies.append(dev_acc)
        train_losses.append(train_loss)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}, train f1 :: {train_f1 :.3f}, dev f1 :: {dev_f1 :.3f}")
    plot(train_accuracies,val_accuracies,train_losses,args.epochs)

def plot(train_accuracy, val_accuracy, train_loss, num_epochs):
    # 第一张图：训练集和验证集准确率
    epochs = np.arange(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # 创建一个1x2的子图网格的第一个子图
    plt.plot(epochs, train_accuracy, label='Training Accuracy', marker=None)
    plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker=None)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 设置横坐标只显示整数
    plt.xticks(epochs)

    # 第二张图：训练集loss
    plt.subplot(1, 2, 2)  # 创建一个1x2的子图网格的第二个子图
    plt.plot(epochs, train_loss, label='Training Loss', marker=None)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 设置横坐标只显示整数
    plt.xticks(epochs)

    # 调整子图间距
    plt.tight_layout()

    # 保存图片
    plt.savefig('./cfimdb_training_validation_plots_lr.png')
    # plt.savefig('./sst_training_validation_plots_lr.png')


def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertSentimentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        
        dev_data = load_data(args.dev, 'valid')
        dev_dataset = SentimentDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = load_data(args.test, 'test')
        test_dataset = SentimentTestDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
        
        dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
        print('DONE DEV')
        test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
        print('DONE Test')
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sent_ids,dev_pred ):
                f.write(f"{p} , {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s  in zip(test_sent_ids,test_pred ):
                f.write(f"{p} , {s} \n")
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="finetune")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")
                                    

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=5e-5)
                  
    args = parser.parse_args()
    return args

def get_parameters(model, model_init_lr, multiplier, classifier_lr):
    parameters = []
    lr = model_init_lr
    before_layer_params = {
        'params': [p for n,p in model.named_parameters()][0:5],
        'lr': lr
    }
    parameters.append(before_layer_params)
    
    for layer in range(12,-1,-1):
        layer_params = {
            'params': [p for n,p in model.named_parameters() if f'bert_layers.{layer}.' in n],
            'lr': lr
        }
        parameters.append(layer_params)
        lr *= multiplier
    
    after_layer_params = {
        'params': [p for n,p in model.named_parameters()][-4:-2],
        'lr': lr
    }
    parameters.append(after_layer_params)

    classifier_params = {
        'params': [p for n,p in model.named_parameters() if 'classify' in n],
        'lr': classifier_lr
    }
    parameters.append(classifier_params)
    return parameters


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    #args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt'

    # print('Training Sentiment Classifier on SST...')
    # config = SimpleNamespace(
    #     filepath='sst-classifier.pt',
    #     lr=args.lr,
    #     use_gpu=args.use_gpu,
    #     epochs=args.epochs,
    #     batch_size=args.batch_size,
    #     hidden_dropout_prob=args.hidden_dropout_prob,
    #     train='data/ids-sst-train.csv',
    #     dev='data/ids-sst-dev.csv',
    #     test='data/ids-sst-test-student.csv',
    #     option=args.option,
    #     dev_out = 'predictions/'+args.option+'-sst-dev-out.csv',
    #     test_out = 'predictions/'+args.option+'-sst-test-out.csv'
    # )

    # train(config)

    # print('Evaluating on SST...')
    # test(config)


    print('Training Sentiment Classifier on cfimdb...')
    config = SimpleNamespace(
        filepath='cfimdb-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-cfimdb-train.csv',
        dev='data/ids-cfimdb-dev.csv',
        test='data/ids-cfimdb-test-student.csv',
        option=args.option,
        dev_out = 'predictions/'+args.option+'-cfimdb-dev-out.csv',
        test_out = 'predictions/'+args.option+'-cfimdb-test-out.csv'
    )

    train(config)

    print('Evaluating on cfimdb...')
    test(config)