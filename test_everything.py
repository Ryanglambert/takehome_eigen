import numpy as np
import os
import pytest

from Vectorize import vectorize

TEST_DOC_DIR = 'test docs'


def test_read_document():
    expected = 'Let me begin'
    output = vectorize._read_document(os.path.join(TEST_DOC_DIR, 'doc1.txt'))[:12]
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


# @pytest.mark.parametrize(
#     "n, expected", [
#         (0, )
#     ]
def test_top_n_words_by_frequency():
    words = np.array(['black', 'blue', 'orange', 'pink', 'red'])
    word_freqs = np.array([2, 1, 20, 10, 3])
    highest_words, highest_word_freqs = vectorize._top_n_words_by_frequency(
        words, word_freqs, 3
    )
    assert isinstance(highest_words, np.ndarray)
    assert list(highest_words) == ['orange', 'pink', 'red']
    assert isinstance(highest_word_freqs, np.ndarray)
    assert list(highest_word_freqs) == [20, 10, 3]


def test_bug_word_frequencies():
    "this bug took me 4 man hours to find sorry bout hte delay"
    # E       assert [3, 10, 8, 9, 2] == [10, 9, 8, 2, 3]
    # E         At index 0 diff: 3 != 10
    # E         Full diff:
    # E         - [3, 10, 8, 9, 2]
    # E         + [10, 9, 8, 2, 3]
    words = np.array(['black', 'blue', 'orange', 'pink', 'red'])
    word_freqs = np.array([ 3,  2,  8,  9, 10])
    highest_words, highest_word_freqs = vectorize._top_n_words_by_frequency(
        words, word_freqs, 5
    )
    assert list(highest_words) == ['red', 'pink', 'orange', 'black', 'blue']
    assert list(highest_word_freqs) == [10, 9, 8, 3, 2]


def test_get_idx_with_word():
    name_lookup = np.array([
        'black', 'red', 'blue', 'orange', 'pink'
    ])
    matrix = np.array([
        [0, 0, 2, 2, 1],
        [1, 1, 3, 1, 4],
        [0, 1, 1, 3, 2],
        [1, 0, 2, 3, 3]
    ])
    word = "black"
    output = vectorize._get_col_mask_with_word(word,
                                               matrix,
                                               name_lookup)
    expected = np.array([False, True, False, True])
    assert output.shape == expected.shape
    assert list(output) == list(expected)



def test_create_hashtag_records():
    docs = [
        'pink orange orange red. pink.',
        'red black blue red red orange. pink red. orange orange.',
        'orange blue red pink. pink pink. red.',
        'orange black red red. red pink pink. pink orange.'
    ]
    doc_names = ['doc1', 'doc2', 'doc3', 'doc4']
    output = vectorize.create_hashtag_records(
        docs, doc_names, max_words=10, min_freq=1, min_doc_freq=1)

    expected = [
        ('red', ["doc1", "doc2", "doc3"], [
            "pink orange orange red.",
            "red black blue red red orange.",
            "pink red."
            "red.",
            "orange black red red.",
            "red pink pink.",
        ]),
        ('pink', ["doc1", "doc2", "doc3"], [
            "pink orange orange red.",
            "pink.",
            "pink red.",
            "orange blue red pink.",
            "pink pink.",
            "red pink pink.",
            "pink orange."
        ]),
        ('orange', ["doc1", "doc2", "doc3"], [
            "pink orange orange red.",
            "red black blue red red orange.",
            "orange orange.",
            "orange blue red pink.",
            "orange black red red.",
            "pink orange"
        ])
    ]


def test_marshal_records():
    unmarshalled = [
        ('red', ["doc1", "doc2", "doc3"], [
            "pink orange orange red.",
            "red black blue red red orange.",
            "pink red."
            "red.",
            "orange black red red.",
            "red pink pink.",
        ]),
        ('pink', ["doc1", "doc2", "doc3"], [
            "pink orange orange red.",
            "pink.",
            "pink red.",
            "orange blue red pink.",
            "pink pink.",
            "red pink pink.",
            "pink orange."
        ]),
        ('orange', ["doc1", "doc2", "doc3"], [
            "pink orange orange red.",
            "red black blue red red orange.",
            "orange orange.",
            "orange blue red pink.",
            "orange black red red.",
            "pink orange"
        ])
    ]
    output = vectorize.marshal_records(unmarshalled)
    expected = [
        ('red', "doc1, doc2, doc3", (
            "pink orange orange red.\n"
            "red black blue red red orange.\n"
            "pink red.\n"
            "red.\n"
            "orange black red red.\n"
            "red pink pink.\n"
        )),
        ('pink', "doc1, doc2, doc3", (
            "pink orange orange red.\n"
            "pink.\n"
            "pink red.\n"
            "orange blue red pink.\n"
            "pink pink.\n"
            "red pink pink.\n"
            "pink orange.\n"
        )),
        ('orange', "doc1, doc2, doc3", (
            "pink orange orange red.\n"
            "red black blue red red orange.\n"
            "orange orange.\n"
            "orange blue red pink.\n"
            "orange black red red.\n"
            "pink orange\n"
        ))
    ]
