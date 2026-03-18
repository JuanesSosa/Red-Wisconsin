from models.tensorflow_arch import build_model

def train_model(X_train, y_train):
    model = build_model(input_dim=30)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    return model, history