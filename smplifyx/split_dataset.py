import os
import shutil
import argparse


def split_dataset(base_dir, train_dir, val_dir, split_ratio):
    # List all .pkl files in the base directory
    all_files = [f for f in os.listdir(base_dir)
                 if f.endswith('.pkl') and os.path.isfile(os.path.join(base_dir, f))]


    # Split into train/val
    split_index = int(len(all_files) * split_ratio)
    train_files = all_files[:split_index]
    val_files = all_files[split_index:]

    # Create output directories if needed
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy files
    for f in train_files:
        shutil.copy2(os.path.join(base_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy2(os.path.join(base_dir, f), os.path.join(val_dir, f))

    print(f"Done! {len(train_files)} files copied to '{train_dir}', {len(val_files)} to '{val_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split pickle dataset into train and validation folders.")
    parser.add_argument('--base_dir', type=str, default='out', help="Directory with the pickle files.")
    parser.add_argument('--split_ratio', type=float, default=0.9, help="Proportion of data for training.")
    parser.add_argument('--train_dir', type=str, help="Directory to store training data.")
    parser.add_argument('--val_dir', type=str, help="Directory to store validation data.")

    args = parser.parse_args()

    # Fallback to subfolders in base_dir if not explicitly given
    train_dir = args.train_dir or os.path.join(args.base_dir, 'train')
    val_dir = args.val_dir or os.path.join(args.base_dir, 'val')

    split_dataset(args.base_dir, train_dir, val_dir, args.split_ratio)
