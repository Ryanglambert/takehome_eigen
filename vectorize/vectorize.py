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


def create_hashtag_table(doc_dir: str, 
                         dest_path: str,
                         max_words: int=10,
                         min_freq: int=3,
                         min_doc_freq: int=2):
    # read_docs
    docs, doc_names = _read_documents(doc_dir)
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



# create_hashtag_table('test docs', "hashtag_table.csv")
