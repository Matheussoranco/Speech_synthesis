"""
Main entry point for the Speech Synthesis project.
Provides CLI for training, inference, and voice cloning.
"""
import argparse
from src import train, infer, clone

def main():
    parser = argparse.ArgumentParser(description="Speech Synthesis and Voice Cloning")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train", help="Train a TTS or voice cloning model")
    train.add_args(train_parser)

    # Inference
    infer_parser = subparsers.add_parser("infer", help="Synthesize speech from text")
    infer.add_args(infer_parser)

    # Clone
    clone_parser = subparsers.add_parser("clone", help="Clone a voice from a sample")
    clone.add_args(clone_parser)

    args = parser.parse_args()
    if args.command == "train":
        train.run(args)
    elif args.command == "infer":
        infer.run(args)
    elif args.command == "clone":
        clone.run(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
