# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from collections import Counter
import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    
    # Count the classes
    label_counter = Counter(labels)
    
    # Find the minority class
    minority_class = min(label_counter, key=label_counter.get)
    
    # Find the indices of sequences belonging to the minority class
    minority_indices = [i for i, label in enumerate(labels) if label == minority_class]
    
    # Sample with replacement from the minority class
    sampled_minority_indices = random.choices(minority_indices, k=label_counter[not minority_class])
    
    # Combine sampled indices with majority class indices
    sampled_indices = [i for i, label in enumerate(labels) if label != minority_class] + sampled_minority_indices
    
    # Get sampled sequences and labels
    sampled_seqs = [seqs[i] for i in sampled_indices]
    sampled_labels = [labels[i] for i in sampled_indices]
    
    # Shuffle the sampled sequences and labels
    combined = list(zip(sampled_seqs, sampled_labels))
    random.shuffle(combined)
    sampled_seqs, sampled_labels = zip(*combined)
    
    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # Dictionary for encoding
    encode_dict = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1]
    }

    # Empty list for the output encodings
    output_encodings = []

    # Loop to translate the strings
    for seq in seq_arr:
        seq_encodings = []
        for nucleotide in seq:
            encodings = encode_dict[nucleotide]
            seq_encodings.append(encodings)
        concat_seqs = np.concatenate(np.array(seq_encodings))    
        output_encodings.append(concat_seqs)

    # Convert to numpy array
    output_encodings = np.array(output_encodings)
    
    return output_encodings
