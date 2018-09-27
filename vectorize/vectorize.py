import csv
import numpy as np
import os
import re

from sklearn.feature_extraction.text import CountVectorizer

SENT = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")


def _read_document(doc_path: str) -> str:
    "reads a *.txt file from a specified location"
    # Memory friendly
    # will speed up later with a generator that 'paginates'
    with open(doc_path, 'r') as f:
        return f.read()


def _read_documents(dir_path: str):
    "read a list of files and output a list of docs and filenames associated"
    doc_names = os.listdir(dir_path)
    docs = []
    for doc_name in doc_names:
        doc = _read_document(os.path.join(dir_path, doc_name))
        docs.append(doc)
    return docs, doc_names


def _flatten(list):
    return [item for sublist in list for item in sublist]


def vectorize(docs, sents):
    """
    Parameters
    ----------
    docs : list
        list of strings

    Returns
    -------
    td_matrix : sp.csr_matrix
        matrix that represents each document by how many times each term appeared in it
    tsent_matrix : sp.csr_matrix
        matrix that represents each by how many times each term appeared in it
    vocab : list
        a list of all of the vocab words that are found in all documents
    """
    vec = CountVectorizer(stop_words='english')
    td_matrix = vec.fit_transform(docs)
    tsent_matrix = vec.transform(sents)
    vocab = np.array(vec.get_feature_names())
    
    return td_matrix, tsent_matrix, vocab


def _get_document_occurences(td_matrix: np.array):
    "count how many documents each word appeared in"
    return np.array(
        (td_matrix != 0).sum(axis=0)
    ).reshape(-1,)


def _get_word_frequencies(td_matrix: np.array):
    "calculate the frequency of each word"
    return np.array(
        td_matrix.sum(axis=0)
    ).reshape(-1,)


def _top_n_words_by_frequency(words, word_freq, n, reverse=True):
    "return top n occuring words and frequencies"

    sign = 1. if not reverse else -1.
    n = word_freq.shape[0] if n > word_freq.shape[0] else n
    
    indices = np.argpartition(
        sign * word_freq, np.arange(n))[:n]
    return words[indices].copy(), word_freq[indices].copy()


def _get_col_mask_with_word(word: np.array,
                            matrix: np.array,
                            lookup: np.array):
    "Get's a mask for the col in `matrix` that corresponds to the location of the word in lookup"
    word_mask = lookup == word
    word_column = matrix.T[word_mask].T
    doc_mask = (word_column > 0)

    # this is a bit hacky but I'm under some time constraints
    # should be fixed by properly dealing with csr vs ndarray
    if not isinstance(doc_mask, np.ndarray):
        doc_mask = doc_mask.toarray()

    doc_mask = doc_mask.reshape(-1,)
    return doc_mask



def create_hashtag_records(docs: list, 
                           doc_names: list,
                           max_words: int=10,
                           min_freq: int=3,
                           min_doc_freq: int=2):
    if not len(doc_names) == len(docs):
        raise ValueError("docs and doc_names should be the same length")
    # read_docs
    sents = _flatten([SENT.split(doc) for doc in docs])
    doc_names, sents = np.array(doc_names), np.array(sents)

    # vectorize text
    td_matrix, tsent_matrix, vocab = vectorize(docs, sents)

    # extract words limited by `num_hashtags` after filtering for atleast `min_hashtag_freq`
    word_doc_occurences = _get_document_occurences(td_matrix)
    word_frequencies = _get_word_frequencies(td_matrix)

    # filter down to words of interest based on min_freq, max_words, and min_doc_freq
    mask_min_doc_freq = word_doc_occurences >= min_doc_freq
    mask_min_word_freq = word_frequencies >= min_freq


    filtered_vocab = vocab[mask_min_doc_freq & mask_min_word_freq].copy()
    filtered_frequencies = word_frequencies[mask_min_doc_freq & mask_min_word_freq].copy()

    # Sort and retrieve the top 'n' where n -> `max_words` 
    # np.argpartition avoids sorting the entire list (good for time!)
    top_n_words, top_n_freqs = _top_n_words_by_frequency(filtered_vocab, filtered_frequencies, n=max_words)

    records = []
    for word, freq in zip(top_n_words, top_n_freqs):
        mask_doc_with_word = _get_col_mask_with_word(
            word,
            td_matrix,
            vocab)
        doc_names_relevant = doc_names[mask_doc_with_word]
        mask_sents_relevant = _get_col_mask_with_word(
            word,
            tsent_matrix,
            vocab)
        sents_relevant = sents[mask_sents_relevant]
        records.append(
            (word, doc_names_relevant, sents_relevant)
        )
    return records
        
        

# create_hashtag_table('test docs', "hashtag_table.csv")
