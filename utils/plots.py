import matplotlib.pyplot as plt


def plot_losses(history):
    plt.figure(figsize=(10,5))
    plt.plot(history["generator"], label="Generator")
    plt.plot(history["disc_static"], label="Disc static")
    plt.plot(history["disc_temporal"], label="Disc temporal")
    plt.legend()
    plt.title("Training losses")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()


def plot_train_val(train, val, title):
    plt.figure(figsize=(8,5))
    plt.plot(train, label="Train")
    plt.plot(val, label="Validation")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_real_fake_scores(real_scores, fake_scores):
    plt.figure(figsize=(8,5))
    plt.hist(real_scores, bins=30, alpha=0.5, label="Real")
    plt.hist(fake_scores, bins=30, alpha=0.5, label="Fake")
    plt.legend()
    plt.title("Discriminator scores")
    plt.show()


def plot_epsilon(eps_history):
    plt.figure(figsize=(8,5))
    plt.plot(eps_history)
    plt.title("Differential Privacy ε over time")
    plt.xlabel("Epoch")
    plt.ylabel("ε")
    plt.show()
