import numpy as np
import tensorflow as tf
from anc2vec.train.train import fit
from anc2vec.train.dataset import Dataset
from anc2vec.train.utils import Tokenizer
from anc2vec.train.onto import Ontology

def test_dataset_loading():
    # Test dataset creation
    go = Ontology('./go.obo', with_rels=True, include_alt_ids=False)
    tok = Tokenizer(go)

    dataset = Dataset(
        tokenizer=tok,
        embeddings_path='./go-full.sbert.ontology.embeddings.npy',
        batch_sz=2,
        buffer_sz=10
    ).build()

    # Inspect the first batch
    for batch in dataset.take(1):
        x, (y, z, x_reconstr) = batch
        print("Input shape:", x.shape)
        print("Neighbor embeddings shape:", y.shape)
        print("Namespace one-hot shape:", z.shape)
        print("Reconstruction target shape:", x_reconstr.shape)
        break

def test_full_training():
    # Test the full training pipeline
    embeddings = fit(
        obo_fin='./go.obo',
        embeddings_path='./go-full.sbert.ontology.embeddings.npy',
        embedding_sz=200,
        batch_sz=32,
        num_epochs=100
    )

    print("Number of learned embeddings:", len(embeddings))
    print("Embedding dimension:", next(iter(embeddings.values())).shape)

    # save to disk
    np.save('./go-full.sbert-anc2vec.embeddings.npy', embeddings)

def test_basic_training():
    # Test the basic training pipeline
    embeddings = fit(
        obo_fin='./go-basic.obo',
        embeddings_path='./go-basic.sbert.ontology.embeddings.npy',
        embedding_sz=200,
        batch_sz=32,
        num_epochs=100
    )

    print("Number of learned embeddings:", len(embeddings))
    print("Embedding dimension:", next(iter(embeddings.values())).shape)

    # save to disk
    np.save('./go-basic.sbert-anc2vec.embeddings.npy', embeddings)

if __name__ == "__main__":
    # print("Testing dataset loading...")
    # test_dataset_loading()

    # print("\nTesting full training pipeline...")
    # test_full_training()

    print("\nTesting basic training pipeline...")
    test_basic_training()
