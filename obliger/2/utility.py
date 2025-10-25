from argparse import ArgumentParser
import numpy as np


def parse_args():
    """Command line arguments."""
    parser = ArgumentParser(description="CLI arguments for regression models")

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output (may be very long)"
    )
    parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for model (default 0.1)"
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default 10)"
    )

    # logreg+
    parser.add_argument(
        "-tol", "--tolerance",
        type=float,
        default=1.0,
        help="Tolerance for early stopping (default 1) (only used in logreg/ovr)"
    )
    parser.add_argument(
        "-p", "--patience",
        type=int,
        default=10,
        help="Epochs of no improvements before stop (default 10) (only used in logreg/ovr)"
    )

    args = parser.parse_args()
    return args
