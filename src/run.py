import basic_net, fancy_net
import util
import sys, random
import numpy as np

from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier

random.seed(20)

drivers_list = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                 'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                 'p075','p081']
nb_drivers = len(drivers_list)
random.shuffle(drivers_list)

def main():
    im_rows, im_cols   = 64, 64
    batch_size         = 64
    nb_epoch           = 15
    random_state       = 51
    colors             = 1
    validation_size    = 2          # 26 total drivers
    n_neighbors        = 5          # Number of neighbors for KNN

    train_data, train_target, driver_id, _ = util.load_train_data(im_rows,
                                                im_cols, colors)

    drivers_list_train = drivers_list[0:nb_drivers-validation_size]

    x_train, y_train, train_index = util.copy_selected_drivers(train_data, \
                                    train_target, driver_id, drivers_list_train)

    drivers_list_valid = drivers_list[nb_drivers-validation_size:nb_drivers]

    x_valid, y_valid, test_index = util.copy_selected_drivers(train_data, \
                                    train_target, driver_id, drivers_list_valid)

    print 'Train: {}    Valid: {}'.format(len(x_train), len(x_valid))
    print 'Train drivers: {} '.format(drivers_list_train)
    print 'Test drivers: {}'.format(drivers_list_valid)

    if sys.argv[1] == "load":
        model = util.read_model()
    elif sys.argv[1] == "basic_v1":
        model = basic_net.Basic_Net_v1(im_rows, im_cols, colors)
    elif sys.argv[1] == "basic_v2":
        model = basic_net.Basic_Net_v2(im_rows, im_cols, colors)
    elif sys.argv[1] == "fancy_v1":
        model = fancy_net.Fancy_Net_v1(im_rows, im_cols, colors)
    elif sys.argv[1] == "fancy_v2":
        model = fancy_net.Fancy_Net_v2(im_rows, im_cols, colors)

    if sys.argv[1] != "load":
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, \
            verbose=1, validation_data=(x_valid, y_valid))
        util.save_model(model, sys.argv[1]+'_'+str(im_rows))

    score = model.evaluate(x_valid, y_valid, verbose=0)
    predictions_valid = model.predict(x_valid, batch_size=128, verbose=1)

    top1 = 0
    for i in range(len(y_valid)):
        if np.argmax(y_valid[i]) == np.argmax(predictions_valid[i]):
            top1 += 1
    top1 /= float(len(y_valid))
    print 'Final log_loss: {}, top 1 accuracy: {}, rows: {} cols: {} epoch: {}'\
            .format(score, top1, im_rows, im_cols, nb_epoch)


    # K-Nearest Neighbors
    interm_layer_model = util.build_interm_model(model)
    interm_train = interm_layer_model.predict(x_train, batch_size=batch_size, \
                                                          verbose=1)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(interm_train, y_train)

    interm_valid = interm_layer_model.predict(x_valid, batch_size=128, verbose=1)
    knn_predictions = knn.predict(interm_valid)

    knn_score = 0
    for i in range(len(y_valid)):
        if np.argmax(y_valid[i]) == np.argmax(knn_predictions[i]):
            knn_score += 1
    knn_score /= float(len(y_valid))
    print 'K Nearest Neighbors accuracy with {} neighbors: {}'\
          .format(n_neighbors, knn_score)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Please enter the net to train"
    else:
        main()
