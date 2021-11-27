import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification


def predict_ner(sentence):
    tokens = tokenizer(sentence, max_length=514, truncation=True)
    ids = tokens.word_ids()[1:-1]

    preds = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0),
                          attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    preds = torch.argmax(preds.logits.squeeze(), axis=1)
    words = tokenizer.batch_decode(tokens['input_ids'])[1:-1]
    value_preds = [label_list[i] for i in preds][1:-1]
    labels = list(set(value_preds))
    if 'O' in labels:
        labels.remove('O')
    if len(labels) == 0:
        return 'noEntity'
    else:
        return labels[0].split('-')[1]


label_list = ['O', 'B-persName-surname', 'B-placeName-settlement', 'B-orgName', 'B-geogName',
              'B-persName-addName', 'B-persName-forename', 'B-persName', 'I-orgName', 'B-time', 'I-time',
              'I-geogName', 'I-persName-addName', 'B-placeName-country', 'B-date', 'I-date', 'I-persName',
              'B-placeName', 'B-placeName-region', 'I-placeName-settlement', 'B-placeName-bloc',
              'I-persName-surname', 'I-persName-forename', 'I-placeName-country', 'B-placeName-district',
              'I-placeName-district', 'I-placeName-region', 'I-placeName-bloc', 'I-placeName']

tokenizer = AutoTokenizer.from_pretrained('best-ner.model/', model_max_length=514)
model = AutoModelForTokenClassification.from_pretrained('./best-ner.model/', num_labels=len(label_list))
klej_test = pd.read_csv('train.tsv', sep='\t', header=0)
x = klej_test['sentence']
# print(len(x))
y = klej_test['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

y_pred = []

for sentence in x_test:
    label = predict_ner(sentence)
    # print(sentence)
    # print(label)
    y_pred.append(label)

# df = pd.DataFrame({'sentence': x_test[:100], 'predicted_tag': y_pred[:100], 'klej tag': y_test[:100]})
# df.to_csv('herbert_on_klej.csv')

print('precision, recall, f1, average=macro')
print(precision_recall_fscore_support(y_test, y_pred, average='macro'))
print('precision, recall, f1, average=micro')
print(precision_recall_fscore_support(y_test, y_pred, average='micro'))
print('precision, recall, f1, average=weighted')
print(precision_recall_fscore_support(y_test, y_pred, average='weighted'))
print('accuracy: ', accuracy_score(y_test, y_pred))
