from sklearn.model_selection import train_test_split
import pandas as pd


def load_file(data_file_path):
    ''' Function for loading data from .iob files or file with indices to such files
    :param data_file_path: path to iob
    '''

    x_data, y_data = [], []

    # Get data from iob file
    if data_file_path.endswith('.iob') or data_file_path.endswith('.tsv'):
        x_data, y_data = load_iob(data_file_path)

    return x_data, y_data


def load_iob(file_path, extra_features=False):
    """Loads data and label from a file.

    Args:
        file_path (str): path to the file.
        extra_features(bool): use dictionary features from iob

        The file format is tab-separated values.
        A blank line is required at the end of a sentence.

        For example:
        ```
        EU	B-ORG
        rejects	O
        German	B-MISC
        call	O
        to	O
        boycott	O
        British	B-MISC
        lamb	O
        .	O

        Peter	B-PER
        Blackburn	I-PER
        ...
        ```

    Returns:
        tuple(numpy array, numpy array): data and labels.

    Example:
         filename = 'conll2003/en/ner/train.txt'
         data, labels = load_data_and_labels(filename)
    """
    sents, labels = [], []
    words, tags = [], []
    with open(file_path, 'r') as f:
        for line in f:
            if "DOCSTART" in line:
                continue
            line = line.rstrip()
            if line:
                cols = line.split('\t')
                if extra_features:
                    words.append([cols[0]] + cols[3:-1])
                else:
                    words.append(cols[0])
                tags.append(cols[-1])
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], []
        return sents, labels


def transform_labels(labels_list):
    new_labels = []
    for label in labels_list:
        if label == 'O':
            new_labels.append(label)
        elif '#' in label:
            new_labels.append(label.split('#')[-1])
        else:
            new_labels.append(label)
    return new_labels


def write_data_to_file(file_name, x, y):
    tokens = []
    labels = []
    for i in range(len(x)):
        for j in range(len(x[i])):

            tokens.append(x[i][j])
            labels.append(y[i][j])

    data = {'tokens': tokens, 'labels': labels}
    df = pd.DataFrame(data)
    print(df.head())

    df.to_csv(file_name, header=False, index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x, y = load_file('nkjp-nested-simplified-v2.iob')
    all_labels = []
    for l in y:
        all_labels.extend(l)
    y_cut = [transform_labels(labels) for labels in y]
    x_train, x_rem, y_train, y_rem = train_test_split(x, y_cut, test_size=0.2, random_state=0)
    x_test, x_val, y_test, y_val = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)

    write_data_to_file('train.csv', x_train, y_train)
    write_data_to_file('dev.csv', x_val, y_val)
