from training.trainer import train_and_save_model


def train_model() -> None:
    auc, model_path = train_and_save_model()
    print(f"Model trained successfully. AUC: {auc:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_model()

