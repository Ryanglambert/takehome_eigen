import numpy as np
import os
import pytest


from Vectorize import vectorize


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DOC_PATH = os.path.join(CUR_DIR, 'test docs')


def test_read_document():
    expected = 'Let me begin'
    output = vectorize._read_document(os.path.join(TEST_DOC_PATH, 'doc1.txt'))[:12]
    assert expected == output


def test_get_doc_occurences():
    array = np.array([
        # words
        [0, 0, 1, 2], # doc 1
        [3, 1, 0, 0], # doc 1
        [3, 2, 1, 0] # doc 1
    ])
    expected = np.array([2, 2, 2, 1])
    output = vectorize._get_document_occurences(array)
    assert list(output) == list(expected)


def test_get_word_occurences():
    array = np.array([
        [0, 0, 1, 2],
        [3, 1, 0, 0],
        [3, 2, 1, 0],
    ])
    expected = np.array([6, 3, 2, 2])
    output = vectorize._get_word_frequencies(array)
    assert list(expected) == list(output)


def test_vectorize():
    documents = [
        'red red red red red red blue blue orange orange',
        'pink pink pink pink'
    ]
    sents = [
        "red red blue",
        "blue blue pink",
        "orange orange red"
    ]
    td_matrix, tsent_matrix, vocab = vectorize.vectorize(
        documents, sents
    )

    assert list(vocab) == ['blue', 'orange', 'pink', 'red']

    assert list(td_matrix.toarray()[0]) == [2, 2, 0, 6]
    assert list(td_matrix.toarray()[1]) == [0, 0, 4, 0]

    assert list(tsent_matrix.toarray()[0]) == [1, 0, 0, 2]
    assert list(tsent_matrix.toarray()[1]) == [2, 0, 1, 0]
    assert list(tsent_matrix.toarray()[2]) == [0, 2, 0, 1]
