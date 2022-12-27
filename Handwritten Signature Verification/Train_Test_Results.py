from Preprocessing import *
from Model_Architecture import *
from datetime import time
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def Test(test_Triplets, siamese_model, batch_size=256):
    pos_scores, neg_scores = [], []

    for data in get_batch(test_Triplets, "Test", batch_size=batch_size):
        prediction = siamese_model.predict(data)
        pos_scores += list(prediction[0])
        neg_scores += list(prediction[1])

    accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
    ap_mean = np.mean(pos_scores)
    an_mean = np.mean(neg_scores)
    ap_stds = np.std(pos_scores)
    an_stds = np.std(neg_scores)

    print(f"Accuracy on test = {accuracy:.5f}")
    return accuracy, ap_mean, an_mean, ap_stds, an_stds


def Train(train_Triplets, test_Triplets, siamese_model):
    save_all = False
    epochs = 15
    batch_size = 128
    max_acc = 0
    train_loss = []
    test_metrics = []

    for epoch in range(1, epochs + 1):
        t = time.time()
        # Training the model on train data
        epoch_loss = []
        for data in get_batch(train_Triplets, "Train", batch_size=batch_size):
            loss = siamese_model.train_on_batch(data)
            epoch_loss.append(loss)
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        train_loss.append(epoch_loss)

        print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time() - t)} sec)")
        print(f"Loss on train    = {epoch_loss:.5f}")

        # Testing the model on test data
        metric = Test(test_Triplets, siamese_model, batch_size=batch_size)
        test_metrics.append(metric)
        accuracy = metric[0]

        # Saving the model weights
        if save_all or accuracy >= max_acc:
            siamese_model.save_weights("siamese_model")
            max_acc = accuracy

    # Saving the model after all epochs run
    siamese_model.save_weights("siamese_model-final")
    return siamese_model


def Extract_encoder(model):
    encoder = get_encoder((128, 128, 3))
    i = 0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i += 1
    return encoder


def Classify_Images(encoder, anchor_image, query_image, threshold=1.3):
    # Getting the encodings for the passed faces
    tensor1 = encoder.predict(anchor_image)
    tensor2 = encoder.predict(query_image)

    distance = np.sum(np.square(tensor1 - tensor2), axis=-1)
    prediction = np.where(distance <= threshold, 0, 1)
    return prediction


def Model_Metrics(pos_list, neg_list):
    true = np.array([0] * len(pos_list) + [1] * len(neg_list))
    prediction = np.append(pos_list, neg_list)

    # Compute and print the accuracy
    print(f"\nAccuracy of model: {accuracy_score(true, prediction)}\n")

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(true, prediction)

    categories = ['Similar', 'Different']
    names = ['True Similar', 'False Similar', 'False Different', 'True Different']
    percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='', xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)


def Classification(test_Triplets, encoder):
    pos_list = np.array([])
    neg_list = np.array([])

    for data in get_batch(test_Triplets, "Test", batch_size=256):
        a, p, n = data
        pos_list = np.append(pos_list, Classify_Images(encoder, a, p))
        neg_list = np.append(neg_list, Classify_Images(encoder, a, n))
        break

    Model_Metrics(pos_list, neg_list)
