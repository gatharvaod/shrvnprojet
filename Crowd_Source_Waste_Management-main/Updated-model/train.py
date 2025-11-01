import argparse
)


class_names = train_ds.class_names
num_classes = len(class_names)
print('Classes:', class_names)


# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
model.summary()


models_dir = Path('models')
models_dir.mkdir(exist_ok=True)


checkpoint_path = models_dir / 'waste_classifier.h5'
cb_ckpt = ModelCheckpoint(str(checkpoint_path), save_best_only=True, monitor='val_accuracy')
cb_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy')


epochs = args.epochs


if args.fast:
# tiny synthetic training loop for sanity checks
print('Running in FAST demo mode (1 epoch, small subset).')
history = model.fit(train_ds.take(5), validation_data=val_ds.take(2), epochs=1, callbacks=[cb_ckpt])
else:
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cb_ckpt, cb_stop])


# Load best model (checkpoint)
best_model = tf.keras.models.load_model(checkpoint_path)


# Evaluate
val_images = []
val_labels = []
for batch_images, batch_labels in val_ds:
preds = best_model.predict(batch_images)
val_images.append(preds)
val_labels.append(batch_labels.numpy())


# For a classification report, gather true/pred label indices
y_true = []
y_pred = []
for batch_images, batch_labels in val_ds:
y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
p = best_model.predict(batch_images)
y_pred.extend(np.argmax(p, axis=1))


print('\nClassification report on validation set:')
print(classification_report(y_true, y_pred, target_names=class_names))


# Save labels
save_labels(class_names, models_dir / 'labels.json')


# Convert to TFLite
tflite_path = models_dir / 'waste_classifier.tflite'
convert_to_tflite(best_model, tflite_path)


print('\nTraining + conversion complete. Models saved to ./models/')




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory containing train/ and val/')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--fast', action='store_true', help='Run a very short demo training for pipeline testing')
args = parser.parse_args()
main(args)