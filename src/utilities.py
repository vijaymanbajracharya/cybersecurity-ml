def categorical_to_binary(labels):
    binary_classes = []
    for label in labels:
        if label[0] == 'Normal':
            binary_classes.append(0)
        else:
            binary_classes.append(1)
    return binary_classes