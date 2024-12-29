Op_labels = [0, 1, 3]
Op_labels_categorical = to_categorical(Op_labels, num_classes=4)
print(Op_labels_categorical)