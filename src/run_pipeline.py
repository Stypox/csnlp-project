#!/usr/bin/env python
"""
Simple script to run the C4 pipeline with script.py
"""

import argparse
from c4_pipeline import C4Pipeline

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run script.py with C4 dataset")
    parser.add_argument("--layer", type=int, default=19, help="Layer to apply gradient regularization")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--reg_lambda", type=float, default=1e-2, help="Regularization lambda")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--save_path", type=str, default="./models/model.pt", help="Path to save the model")
    args = parser.parse_args()
    
    # Create and run the pipeline
    pipeline = C4Pipeline()
    pipeline.run(
        layer=args.layer,
        learning_rate=args.learning_rate,
        reg_lambda=args.reg_lambda,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_steps=args.steps,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main() 