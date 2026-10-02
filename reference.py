import os
import glob
import pickle
import random
import argparse

import numpy as np
from PIL import Image

def sigmoid(net):
    net = np.clip(net, -500, 500)
    return 1.0 / (1.0 + np.exp(-net))


def sigmoid_derivative(a):
    return a * (1.0 - a)

class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output=1, learning_rate=0.1, seed=42):
        rng = np.random.default_rng(seed)


        self.W1 = rng.uniform(-0.5, 0.5, (n_hidden, n_input))
        self.b1 = rng.uniform(-0.5, 0.5, (n_hidden,))

        self.W2 = rng.uniform(-0.5, 0.5, (n_output, n_hidden))
        self.b2 = rng.uniform(-0.5, 0.5, (n_output,))

        self.alpha = learning_rate

    def forward(self, x):
        net_hidden = self.W1 @ x + self.b1
        a_hidden = sigmoid(net_hidden)

        net_out = self.W2 @ a_hidden + self.b2
        a_out = sigmoid(net_out)

        return a_hidden, a_out

    @staticmethod
    def error(t, a_out):
        return 0.5 * np.sum((t - a_out) ** 2)

    def backward(self, x, t, a_hidden, a_out):
        delta_out = (t - a_out) * sigmoid_derivative(a_out)
        delta_hidden = sigmoid_derivative(a_hidden) * (self.W2.T @ delta_out)

        self.W2 += self.alpha * np.outer(delta_out, a_hidden)
        self.b2 += self.alpha * delta_out
        self.W1 += self.alpha * np.outer(delta_hidden, x)
        self.b1 += self.alpha * delta_hidden

    def train_step(self, x, t):
        a_hidden, a_out = self.forward(x)
        e = self.error(t, a_out)
        self.backward(x, t, a_hidden, a_out)
        return e

    def predict_prob(self, x):
        _, a_out = self.forward(x)
        return a_out

    def predict_label(self, x, threshold=0.5):
        return 1 if self.predict_prob(x)[0] >= threshold else 0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2, "alpha": self.alpha}, f)

    def load(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]
        self.alpha = d["alpha"]

def load_image_as_vector(path, img_size):
    img = Image.open(path).convert("L")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr.flatten()

def build_dataset(cats_dir, dogs_dir, img_size):
    X, y = [], []
    cat_files = sorted(glob.glob(os.path.join(cats_dir, "*.jpg")) + glob.glob(os.path.join(cats_dir, "*.jpeg")))
    dog_files = sorted(glob.glob(os.path.join(dogs_dir, "*.jpg")) + glob.glob(os.path.join(dogs_dir, "*.jpeg")))

    for f in cat_files:
        try:
            X.append(load_image_as_vector(f, img_size))
            y.append(0)
        except Exception as e:
            print(f"[skip] could not read {f}: {e}")

    for f in dog_files:
        try:
            X.append(load_image_as_vector(f, img_size))
            y.append(1)
        except Exception as e:
            print(f"[skip] could not read {f}: {e}")

    if not X:
        raise RuntimeError("No images loaded - check folder paths.")

    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(cat_files)} cat images and {len(dog_files)} dog images ")
    return X, y


def train(net, X, y, epochs, verbose=True):
    n_samples = X.shape[0]
    indices = list(range(n_samples))

    for epoch in range(1, epochs + 1):
        random.shuffle(indices) # reshuffling each epoch
        total_error = 0.0
        correct = 0

        for idx in indices:
            x = X[idx]
            t = np.array([y[idx]], dtype=np.float64)

            total_error += net.train_step(x, t)

            pred = net.predict_label(x)
            if pred == y[idx]:
                correct += 1

        avg_error = total_error / n_samples
        accuracy = correct / n_samples

        if verbose:
            print(f"Epoch {epoch:3d}/{epochs} | avg error E = {avg_error:.4f} " f"| training accuracy = {accuracy*100:.2f}%")

    return net

def evaluate(net, X, y):
    TP = FP = TN = FN = 0

    for i in range(X.shape[0]):
        pred = net.predict_label(X[i])
        actual = y[i]

        if actual == 1 and pred == 1:
            TP += 1
        elif actual == 0 and pred == 1:
            FP += 1
        elif actual == 0 and pred == 0:
            TN += 1
        elif actual == 1 and pred == 0:
            FN += 1

    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)

    print("\nConfusion Matrix")
    print(f"{'':15s}{'pred: cat':>12s}{'pred: dog':>12s}")
    print(f"{'actual: cat':15s}{TN:12d}{FP:12d}")
    print(f"{'actual: dog':15s}{FN:12d}{TP:12d}")

    print(f"\nAccuracy  = {accuracy*100:.2f}%")
    print(f"Precision = {precision*100:.2f}%")
    print(f"Recall    = {recall*100:.2f}%")
    print(f"F1-score  = {f1*100:.2f}%")

    return {"accuracy": accuracy, "precision": precision,"recall": recall, "f1": f1, "TP": TP, "FP": FP, "TN": TN, "FN": FN}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--cats", required=True)
    p_train.add_argument("--dogs", required=True)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--hidden", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=0.1)
    p_train.add_argument("--img_size", type=int, default=32)
    p_train.add_argument("--model", default="model.pkl")

    p_test = sub.add_parser("test")
    p_test.add_argument("--cats", required=True)
    p_test.add_argument("--dogs", required=True)
    p_test.add_argument("--img_size", type=int, default=32)
    p_test.add_argument("--model", default="model.pkl")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--image", required=True)
    p_pred.add_argument("--img_size", type=int, default=32)
    p_pred.add_argument("--model", default="model.pkl")

    args = parser.parse_args()

    if args.mode == "train":
        X, y = build_dataset(args.cats, args.dogs, args.img_size)
        n_input = X.shape[1]
        net = NeuralNetwork(n_input=n_input, n_hidden=args.hidden, n_output=1, learning_rate=args.lr)
        train(net, X, y, epochs=args.epochs)
        net.save(args.model)
        print(f"\nSaved trained model to {args.model}")

    elif args.mode == "test":
        X, y = build_dataset(args.cats, args.dogs, args.img_size)
        n_input = X.shape[1]
        net = NeuralNetwork(n_input=n_input, n_hidden=1)
        net.load(args.model)
        evaluate(net, X, y)

    elif args.mode == "predict":
        x = load_image_as_vector(args.image, args.img_size)
        net = NeuralNetwork(n_input=x.shape[0], n_hidden=1)
        net.load(args.model)
        prob = net.predict_prob(x)[0]
        label = "DOG" if prob >= 0.5 else "CAT"
        print(f"Prediction: {label}  (output activation a = {prob:.4f})")


if __name__ == "__main__":
    main()