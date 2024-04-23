import pandas as pd
import torch
import shutil
import numpy as np
from torcheval.metrics.functional import multiclass_f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.nn.functional as nnf
from sklearn.metrics import classification_report
import seaborn as sn

# load and name the dataset
df = pd.read_csv('/Users/anastasiabelaeva/Desktop/Postgraduate/данные/LLMs literacy/train_val_test_df/df_preprocessed.csv')
target_list = ['score_0', 'score_1', 'score_2']
df.rename(columns={'student_id': 'id'}, inplace=True)

# hyperparameters
MAX_LEN = 50
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-05

# import sbert
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_mt_nlu_ru")

# create a dataset class
class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df['Preprocessed_texts']
        self.targets = self.df[target_list].values
        self.max_len = max_len
        self.ids = self.df.id.values


    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())
        ids = self.ids[index]

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'id': ids
        }

# prepare and load train and test datasets
train_size = 0.7
train_df = df.sample(frac=train_size, random_state=6).reset_index(drop=True)
train_df.set_index('id', inplace=True)
df.set_index('id', inplace=True)
df_test = df[~df.index.isin(train_df.index)]
train_df.reset_index(inplace=True)
df_test.reset_index(inplace=True)
train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
test_dataset = CustomDataset(df_test, tokenizer, MAX_LEN)
TEST_BATCH_SIZE = 32
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=TRAIN_BATCH_SIZE,
                                                shuffle=False,
                                                num_workers=0
                                                )

test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=TEST_BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=0
                                              )

# check if  GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# function to load the checkpoint
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

# function to save the checkpoint
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

# create a Model class
class RUClass(torch.nn.Module):
    def __init__(self):
        super(RUClass, self).__init__()
        self.ru_model = AutoModel.from_pretrained("ai-forever/sbert_large_mt_nlu_ru", return_dict=True, num_labels=3)
        self.dropout = torch.nn.Dropout(0.1)
        self.linear1 = torch.nn.Linear(1024, 3)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.ru_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear1(output_dropout)
        return output

model = RUClass()
model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

# train and test the model
def train_model(n_epochs, training_loader, test_data_loader, model,
                optimizer, checkpoint_path, best_model_path):
    train_loss_min = np.Inf
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        test_loss = 0

        model.train()
        correct_train = 0
        train_f1 = 0
        train_outputs = []
        for batch_idx, data in enumerate(training_loader):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            correct_train += (torch.argmax(outputs, 1) == torch.argmax(targets, 1)).sum() / TRAIN_BATCH_SIZE
            train_f1 += multiclass_f1_score(input=outputs, target=torch.argmax(targets, 1), num_classes=3)
            train_outputs.extend(torch.max(outputs, 1)[1].tolist())
        f1 = train_f1 / len(training_loader)

        print('Epoch: {} \tAverage Training F1: {:.6f}'.format(
            epoch,
            f1))
        with torch.no_grad():
            correct_test = 0
            f1_test = 0
            if epoch == 30:
                ids_extend = []
                test_targets = []
                test_outputs_soft = []
                outputs_proba = []
                test_outputs = []
                for batch_idx, data in enumerate(test_data_loader):
                    ids = data['input_ids'].to(device, dtype=torch.long)
                    mask = data['attention_mask'].to(device, dtype=torch.long)
                    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                    targets = data['targets'].to(device, dtype=torch.float)
                    outputs = model(ids, mask, token_type_ids)

                    loss = loss_fn(outputs, targets)
                    test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.item() - test_loss))
                    correct_test += (torch.argmax(outputs, 1) == torch.argmax(targets, 1)).sum() / TEST_BATCH_SIZE
                    f1_test += multiclass_f1_score(input=outputs, target=torch.argmax(targets, 1), num_classes=3)

                    test_outputs_soft.extend((torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy())
                    test_targets.extend((torch.max(torch.exp(targets), 1)[1]).data.cpu().numpy())
                    ids_extend.extend(data['id'].tolist())
                    outputs_softmax = nnf.softmax(outputs, dim=1)
                    outputs_proba.extend(outputs_softmax)
                    test_outputs.extend(torch.max(outputs, 1)[1].tolist())

                cf_matrix = confusion_matrix(test_targets, test_outputs_soft)
                df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in target_list],
                                     columns=[i for i in target_list])

            train_loss = train_loss / len(training_loader)
            test_loss = test_loss / len(test_data_loader)

            f1_test = f1_test / len(test_data_loader)

            print('Epoch: {} \tAverage Test F1: {:.6f}'.format(
                epoch,
                f1_test))
            print('Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Test Loss: {:.6f}'.format(
                epoch,
                train_loss,
                test_loss
            ))

            checkpoint = {
                'epoch': epoch + 1,
                'train_loss_min': train_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            # save checkpoint
            save_ckp(checkpoint, False, checkpoint_path, best_model_path)
            if train_loss <= train_loss_min:
                print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,
                                                                                                train_loss))
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                train_loss_min = train_loss
    return model, test_outputs, test_targets, df_cm, ids_extend, train_outputs


ckpt_path = "/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/экономыш 2023/checkpoints/curr_ckpt"
best_model_path = "/Users/anastasiabelaeva/Desktop/Postgraduate/данные/CT/экономыш 2023/best_model.pt"

trained_model, test_output, test_targets, conf_matrix, ids, train_outputs = \
    train_model(EPOCHS, train_data_loader, test_data_loader, model, optimizer, ckpt_path, best_model_path)

# save train and test datasets to use them in Random Forest
df_test['LLM_outputs'] = test_output
df_test.to_csv('/Users/anastasiabelaeva/Desktop/Postgraduate/данные/LLMs literacy/дообучение/test_LLM_outputs.csv')
train_df['LLM_outputs'] = train_outputs
train_df.to_csv('/Users/anastasiabelaeva/Desktop/Postgraduate/данные/LLMs literacy/дообучение/train_LLM_outputs.csv')

# metrics report
report = classification_report(test_targets, test_output)
print(report)
plt.figure(figsize=(10, 7))
sn.heatmap(conf_matrix, annot=True, annot_kws={'size': 15})
print(conf_matrix)
plt.show()


