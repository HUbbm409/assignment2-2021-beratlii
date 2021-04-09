from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np


def read_dataset(dataset_name):
    file = open(dataset_name, "r")
    dataset = file.read().splitlines()
    file.close()

    return dataset


def get_dataset_list(dataset):
    neg_row_list = []
    pos_row_list = []
    test_row_list = []
    true_list = []

    for row in dataset[2400:]:
        row_list = row.split()
        token = ' '.join([str(elem) for elem in row_list[3:]])
        if row_list[1] == "neg":
            neg_row_list.append(token)
        else:
            pos_row_list.append(token)

    for row in dataset[:2400]:
        row_list = row.split()
        token = ' '.join([str(elem) for elem in row_list[3:]])
        test_row_list.append(token)
        true_list.append(row_list[1])

    return neg_row_list, pos_row_list, test_row_list, true_list


def vectorizer_for_rows(pos_row_list, neg_row_list, bow_gram, stop_words):
    pos_word_vectorizer = TfidfVectorizer(ngram_range=(bow_gram, bow_gram), stop_words=stop_words, analyzer='word')
    pos_word_matrix = pos_word_vectorizer.fit_transform(pos_row_list)
    pos_word_vocabulary = pos_word_vectorizer.vocabulary_

    neg_word_vectorizer = TfidfVectorizer(ngram_range=(bow_gram, bow_gram), stop_words=stop_words, analyzer='word')
    neg_word_matrix = neg_word_vectorizer.fit_transform(neg_row_list)
    neg_word_vocabulary = neg_word_vectorizer.vocabulary_

    return pos_word_vectorizer, pos_word_matrix, pos_word_vocabulary, neg_word_vectorizer, neg_word_matrix, neg_word_vocabulary


def get_histogram_for_word_type(pos_word_dict, neg_word_dict, word_number, word_type):
    pos_dict = {}

    for (item, value) in pos_word_dict.items():
        if item not in neg_word_dict.keys():
            pos_dict[item] = value

    pos_list = list(pos_dict.items())
    ordered_pos_list = sorted(pos_list, key=lambda l: l[1], reverse=True)

    neg_dict = {}
    for item, value in neg_word_dict.items():
        if item not in pos_word_dict.keys():
            neg_dict[item] = value

    neg_list = list(neg_dict.items())
    ordered_neg_list = sorted(neg_list, key=lambda l: l[1], reverse=True)

    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []

    if word_type == "presence":

        for x in range(word_number):
            pos_x.append(ordered_pos_list[x][0])
            pos_y.append(ordered_pos_list[x][1])
        r = pd.DataFrame({"Word/WordPairs": pos_x, 'Frequency': pos_y})
        print(r)

    if word_type == "absence":

        for x in range(word_number):
            neg_x.append(ordered_neg_list[x][0])
            neg_y.append(ordered_neg_list[x][1])
        f = pd.DataFrame({"Word/WordPairs": neg_x, 'Frequency': neg_y})
        print(f)


def get_frequency_by_row_type(pos_word_matrix, pos_word_id_dict, neg_word_matrix, neg_word_id_dict):
    pos_word_frequencies = pos_word_matrix.sum(axis=0)
    pos_words_freq = [(word, pos_word_frequencies[0, idx]) for word, idx in pos_word_id_dict.items()]

    neg_word_frequencies = neg_word_matrix.sum(axis=0)
    neg_words_freq = [(word, neg_word_frequencies[0, idx]) for word, idx in neg_word_id_dict.items()]

    pos_word_dict = dict(pos_words_freq)
    neg_word_dict = dict(neg_words_freq)

    return pos_word_dict, neg_word_dict


def naive_bayes(pos_word_vectorizer, pos_word_matrix, pos_frequency_dict, neg_vectorizer, neg_word_matrix,
                neg_frequency_dict, test_line_list, bow_gram, stop_words):

    pos_words_set = pos_word_vectorizer.get_feature_names()
    neg_words_set = neg_vectorizer.get_feature_names()

    pos_words_count = pos_word_matrix.sum()
    neg_words_count = neg_word_matrix.sum()

    total_words_count = pos_words_count + neg_words_count

    p_pos = pos_words_count / total_words_count
    p_neg = neg_words_count / total_words_count

    uniq_words = set(pos_words_set + neg_words_set)
    uniq_words_count = len(uniq_words)
    # computing bayes theorem with laplace smoothing #
    test_classifier_result = []
    for line in test_line_list:
        test_vectorizer = CountVectorizer(ngram_range=(bow_gram, bow_gram), stop_words=stop_words, analyzer='word')
        line_list = []
        line_list.append(line)
        test_word_matrix = test_vectorizer.fit_transform(line_list)

        probabilty_pos = 0
        probabilty_neg = 0

        for item, value in test_vectorizer.vocabulary_.items():
            probabilty_pos += test_word_matrix.toarray()[0, value] * np.log10((pos_frequency_dict.get(item, 0) + 1) / (
                    pos_words_count + uniq_words_count))
            probabilty_neg += test_word_matrix.toarray()[0, value] * np.log10((neg_frequency_dict.get(item, 0) + 1) / (
                    neg_words_count + uniq_words_count))

        probabilty_pos = probabilty_pos + np.log10(p_pos)
        probabilty_neg = probabilty_neg + np.log10(p_neg)

        if probabilty_pos > probabilty_neg:
            test_classifier_result.append("pos")
        elif probabilty_pos < probabilty_neg:
            test_classifier_result.append("neg")
        elif p_pos >= p_neg:
            test_classifier_result.append("pos")
        else:
            test_classifier_result.append("neg")
    return test_classifier_result


def get_accuracy(true_list, classifier_list):
    correct_prediction = 0
    for index in range(len(true_list)):
        if true_list[index] == classifier_list[index]:
            correct_prediction = correct_prediction + 1

    accuracy = (correct_prediction / len(true_list)) * 100

    print("#########################################################")
    print("             Accuracy: ", accuracy)
    print("#########################################################")


def main(dataset_name, bow_gram, stop_words, word_type):
    dataset = read_dataset(dataset_name)

    neg_row_list, pos_row_list, test_row_list, true_list = get_dataset_list(dataset)

    pos_word_vectorizer, pos_word_matrix, pos_word_vocabulary, neg_word_vectorizer, neg_word_matrix, neg_word_vocabulary = vectorizer_for_rows(
        pos_row_list, neg_row_list, bow_gram, stop_words)

    pos_word_frequency_dict, neg_word_frequency_dict = get_frequency_by_row_type(pos_word_matrix, pos_word_vocabulary,
                                                                       neg_word_matrix,
                                                                       neg_word_vocabulary)

    get_histogram_for_word_type(pos_word_frequency_dict, neg_word_frequency_dict, 10, word_type)

    classifier_list = naive_bayes(pos_word_vectorizer, pos_word_matrix, pos_word_frequency_dict, neg_word_vectorizer,
                                  neg_word_matrix,
                                  neg_word_frequency_dict, test_row_list, bow_gram, stop_words)

    get_accuracy(true_list, classifier_list)


if __name__ == '__main__':
    dataset_name = 'all_sentiment_shuffled.txt'
    bow_gram = 2
    stop_words = "english"
    word_type = "absence"

    main(dataset_name, bow_gram, stop_words, word_type)

