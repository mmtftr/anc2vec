import argparse
import wandb
import numpy as np
from .train import fit

def main():
    parser = argparse.ArgumentParser(description='Train the Anc2Vec model')

    # Required arguments
    parser.add_argument('--obo-file', required=True, help='Path to the OBO file')
    parser.add_argument('--embeddings-path', required=True, help='Path to input embeddings')
    parser.add_argument('--output-path', required=True, help='Path to save output embeddings')

    # Optional arguments
    parser.add_argument('--embedding-size', type=int, default=200, help='Size of the embedding vectors')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')

    # Loss weights
    parser.add_argument('--ance-weight', type=float, default=1.0, help='Weight for ANCE loss')
    parser.add_argument('--name-weight', type=float, default=1.0, help='Weight for namespace loss')
    parser.add_argument('--auto-weight', type=float, default=1.0, help='Weight for autoencoder loss')

    # Learning rate settings
    parser.add_argument('--initial-lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--use-lr-schedule', action='store_true', help='Enable learning rate scheduling')

    # Wandb settings
    parser.add_argument('--wandb-project', default='anc2vec', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', help='Weights & Biases entity/username')

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )

    # Configure loss weights
    loss_weights = {
        'ance': args.ance_weight,
        'name': args.name_weight,
        'auto': args.auto_weight
    }

    # Train model
    embeddings = fit(
        obo_fin=args.obo_file,
        embeddings_path=args.embeddings_path,
        embedding_sz=args.embedding_size,
        batch_sz=args.batch_size,
        num_epochs=args.epochs,
        loss_weights=loss_weights,
        use_lr_schedule=args.use_lr_schedule,
        initial_lr=args.initial_lr
    )

    # Save results
    np.save(args.output_path, embeddings)

    wandb.finish()

if __name__ == '__main__':
    main()