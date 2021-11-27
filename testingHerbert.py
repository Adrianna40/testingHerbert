import nltk.tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from process_poleval import get_poleval_dict
import json


def predict_ner(sentence):
    tokens = tokenizer(sentence, max_length=514, truncation=True)
    ids = tokens.word_ids()[1:-1]

    preds = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0),
                          attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    preds = torch.argmax(preds.logits.squeeze(), axis=1)
    words = tokenizer.batch_decode(tokens['input_ids'])[1:-1]
    value_preds = [label_list[i] for i in preds][1:-1]
    word_preds = []
    sentence_list = []

    for i in range(len(value_preds)):
        if i == 0:
            word_preds.append(value_preds[i])
        elif ids[i - 1] != ids[i]:
            word_preds.append(value_preds[i])

    for i in range(len(words)):
        if i == len(words) - 1:
            sentence_list.append(words[i])
        elif ids[i] == ids[i + 1]:
            words[i + 1] = words[i] + words[i + 1]
        else:
            sentence_list.append(words[i])

    return sentence_list, word_preds


def create_json(my_file, source_file):
    list_of_dict = []
    with open(source_file, 'r', encoding='UTF-8') as tests_json:
        texts = json.load(tests_json)
        for text in texts:
            sentences = []
            labels = []

            for s in nltk_tokenizer.tokenize(text['text']):
                # print(s)
                sentence, label = predict_ner(s)
                sentences.append(sentence)
                labels.append(label)

            herbert_dict = get_poleval_dict(text['id'], text['text'], sentences, labels)
            print(herbert_dict["answers"])
            entities = []
            for label in labels:
                label = list(set(label))
                if 'O' in label:
                    label.remove('O')

                entities.extend(label)
            print(entities)
            list_of_dict.append(herbert_dict)

#    with open(my_file, 'w') as herbert_file:
#        json.dump(list_of_dict, herbert_file)


label_list = ['O', 'B-persName-surname', 'B-placeName-settlement', 'B-orgName', 'B-geogName',
              'B-persName-addName', 'B-persName-forename', 'B-persName', 'I-orgName', 'B-time', 'I-time',
              'I-geogName', 'I-persName-addName', 'B-placeName-country', 'B-date', 'I-date', 'I-persName',
              'B-placeName', 'B-placeName-region', 'I-placeName-settlement', 'B-placeName-bloc',
              'I-persName-surname', 'I-persName-forename', 'I-placeName-country', 'B-placeName-district',
              'I-placeName-district', 'I-placeName-region', 'I-placeName-bloc', 'I-placeName']

tokenizer = AutoTokenizer.from_pretrained('best-ner.model/', model_max_length=514)
model = AutoModelForTokenClassification.from_pretrained('./best-ner.model/', num_labels=len(label_list))
# sentence = 'Adrianna Klank jedzie nad Zatokę Gdańską.'

# print(predict_ner(sentence))


nltk_tokenizer = nltk.data.load('tokenizers/punkt/polish.pickle')
create_json('poleval_herbert_results.json', 'poleval_test_ner_2018.json')


