"""
This file contains the functions for encoding the code into a numeric representation for ML.
"""

import gensim
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sctokenizer import CppTokenizer
from pandas import DataFrame


def cpp_tokenize(code_samples) -> list[list]:
    """
    Tokenize the C++ code samples using the CppTokenizer

    Parameters:
        code_samples (list): list of plain text code samples
    Returns:
        tokenized_samples (list): A list of tokenized code samples
    """
    tokenizer = CppTokenizer()
    
    tokenized_samples = [tokenizer.tokenize(code) for code in code_samples ]
    return tokenized_samples

def token_to_string(tokenized_code) -> list[str]:
    """
    Convert the SCT tokenized code samples to strings
    Parameters:
        tokenized_code (list): list of SCT tokenized code samples
    Returns:
        code_strings (list): A list of strings
    """
    str_tokens = []
    ## Convert the sctokenizer Token objects to strings
    for code in tokenized_code:
        tokens = [token.token_value for token in code]
        str_tokens.append(tokens)
    return str_tokens


def create_tagged_documents(str_tokens) -> list[TaggedDocument]:
    """
    Create a tagged document for the Doc2Vec model

    Parameters:
        tokenized_code (list): list of tokenized code samples
    Returns:
        tagged_documents (list): A list of TaggedDocuments (Doc2Vec format)
    """
    return [TaggedDocument(str_tokens[i], [i]) for i in range(len(str_tokens))]

def train_doc2vec_model(tagged_documents, file_name, vector_size=300, window=40, min_count=1, workers=4, epochs=100) -> Doc2Vec:
    """
    Train a Doc2Vec model on the tagged documents and save it to a file or load it if it already exists
    
    Parameters:
        tagged_documents (list): list of tagged documents
        file_name (str): name of the file to save the model
        vector_size (int): size of the vectors
        window (int): window size
        min_count (int): minimum count of a word to be included in the vocabulary
        workers (int): number of workers
        epochs (int): number of epochs
    Returns:
        model (Doc2Vec): trained Doc2Vec model
    """
    ## do not train if file already exists
    try:
        model = Doc2Vec.load(file_name + ".model")
        return model
    except:
        model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        model.build_vocab(tagged_documents)
        model.train(tagged_documents, total_examples=model.corpus_count, epochs=epochs)
        model.save(file_name + ".model")
        return model
    
def encode(tokenized_code_samples, model) -> DataFrame:
    """
    Encode the tokenized code samples using the Doc2Vec model

    Parameters:
        tokenized_code_samples (list): list of tokenized code samples
        model (Doc2Vec): trained Doc2Vec model
    Returns:
        encoded_data (DataFrame): A DataFrame containing the encoded data
    """
    encoded_data = [model.infer_vector(tokenized_code) for tokenized_code in tokenized_code_samples]
    return DataFrame(encoded_data)