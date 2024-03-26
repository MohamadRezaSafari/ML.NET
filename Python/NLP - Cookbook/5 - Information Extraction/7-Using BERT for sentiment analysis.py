import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from Chapter04.svm_classification import split_dataset
from Chapter04.twitter_sentiment import read_existing_file, clean_data, plot_model



batch_size = 32
DATASET_SIZE = 4000
english_twitter = "Chapter04/twitter_english.csv"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
max_length = 200


def encode_data(df):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    for index, row in df.iterrows():
        tweet = row['tweet']
        label = row['sentiment']
        tokenized = tokenizer.tokenize(tweet)
        bert_input = tokenizer.encode_plus(tweet, add_special_tokens = True,
            max_length = max_length,
            pad_to_max_length = True,
            return_attention_mask = True,)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input[
        'token_type_ids'])
        attention_mask_list.append(bert_input[
        'attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list,
        token_type_ids_list,
        label_list)).map(map_inputs_to_dict)


def map_inputs_to_dict(input_ids, attention_masks,
    token_type_ids, label):
    return {
    "input_ids": input_ids,
    "token_type_ids": token_type_ids,
    "attention_mask": attention_masks,
    }, label


def prepare_dataset(df):
    df = clean_data(df)
    df = pd.concat([df.head(int(DATASET_SIZE/2)),
    df.tail(int(DATASET_SIZE/2))])
    df = df.sample(frac = 1)
    ds = encode_data(df)
    return ds


def fine_tune_model(ds, export_dir):
    (train_dataset, test_dataset, val_dataset) = get_test_train_val_datasets(ds)
    learning_rate = 2e-5
    number_of_epochs = 3
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate,
    epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss,
    metrics=[metric])
    bert_history = model.fit(train_dataset,
    epochs=number_of_epochs,
    validation_data=val_dataset)
    model.save_pretrained(export_dir)
    return model


df = read_existing_file(english_twitter)
dataset = prepare_dataset(df)
model = fine_tune_model(dataset,'Chapter05/bert_twitter_model')


def test_new_example(model_path, tweet):
    model = load_existing_model(model_path)
    bert_input = encode_example(tweet)
    tf_output = model.predict([bert_input['input_ids'],
    bert_input['token_type_ids'],
    bert_input['attention_mask']])[0]
    tf_pred = tf.nn.softmax(tf_output,
        axis=1).numpy()[0]
    new_label = np.argmax(tf_pred, axis=-1)
    print(new_label)
    return new_label


test_new_example('Chapter04/bert_twitter_test_model',
"I hate going to school")


def evaluate_model(model, X_test, y_test):
    y_pred = []
    for tweet in X_test:
    bert_input = encode_example(tweet)
    tf_output = model.predict([bert_input['input_ids'],
    bert_input['token_type_ids'],
    bert_input['attention_mask']])[0]
    tf_pred = tf.nn.softmax(tf_output,
    axis=1).numpy()[0]
    new_label = np.argmax(tf_pred, axis=-1)
    y_pred.append(new_label)
    print(classification_report(y_test, y_pred,
    labels=[0, 1], target_names=['negative','positive']))


def encode_example(input_text):
    tokenized = tokenizer.tokenize(input_text)
    bert_input = tokenizer.encode_plus(input_text,
        add_special_tokens = True,
        max_length = max_length,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors='tf')
    return bert_input


def load_and_evaluate_existing_model(export_dir):
    model = load_existing_model(export_dir)
    df = read_existing_file(english_twitter)
    df = clean_data(df)
    df = pd.concat([df.head(200),df.tail(200)])
    (X_train, X_test, y_train, y_test) = split_dataset(df, 'tweet', 'sentiment')
    evaluate_model(model, X_test, y_test)


load_and_evaluate_existing_model(filename)
