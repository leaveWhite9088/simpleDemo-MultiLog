import torch
import argparse
import os
from MultiLog.multilog import MultiLog, MultiLogTrainer
from data.dataset import create_dataloader


def main():
    parser = argparse.ArgumentParser(description='MultiLog: Multivariate Log-based Anomaly Detection')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes (1 for Single2Single)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for LSTM')
    parser.add_argument('--window_size', type=int, default=5, help='Time window size in seconds')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Path to save model')
    parser.add_argument('--sample_ratio', type=float, default=0.01, help='Ratio of data to use (default: 0.05 for 5%)')
    
    args = parser.parse_args()
    
    print(f"MultiLog Model Training")
    print(f"Device: {args.device}")
    print(f"Number of nodes: {args.num_nodes}")
    print(f"Dataset: Single2Single")
    print(f"Data sample ratio: {args.sample_ratio*100:.1f}%")
    print("-" * 50)
    
    # Create data loaders
    train_loader = create_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        mode='train',
        sample_ratio=args.sample_ratio
    )
    
    val_loader = create_dataloader(
        args.data_path,
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        mode='val',
        sample_ratio=args.sample_ratio
    )
    
    # Initialize model
    model = MultiLog(
        num_nodes=args.num_nodes,
        hidden_size=args.hidden_size,
        window_size=args.window_size
    )
    
    # Initialize trainer
    trainer = MultiLogTrainer(model, learning_rate=args.learning_rate, device=args.device)
    
    # Train autoencoder first
    trainer.train_autoencoder(train_loader, epochs=5)
    
    # Train the full model
    print("\nTraining full MultiLog model...")
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        # Training
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Validation
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.save_path, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.save_path, 'best_model.pth'))
            print(f"  Saved best model with validation accuracy: {val_acc:.4f}")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
