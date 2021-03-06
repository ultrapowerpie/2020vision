import basic_net, fancy_net
import util
import sys, random
import numpy as np

from sklearn.metrics import log_loss

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
    nb_epoch           = 10
    random_state       = 51
    colors             = 1
    validation_size    = 2          # 26 total drivers
    nb_models          = 10
    dropout            = 0

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

    models = []
    predictions = np.zeros((len(y_valid), 10))
    for i in range(nb_models):

        if sys.argv[1] == "load":
            if len(sys.argv) < 3:
                print "Please enter the name of the model to load"
            else:
                models.append(util.read_model(sys.argv[2]+'_'+str(i)))
        elif sys.argv[1] == "basic_v1":
            models.append(basic_net.Basic_Net_v1(im_rows, im_cols, colors))
        elif sys.argv[1] == "basic_v2":
            models.append(basic_net.Basic_Net_v2(im_rows, im_cols, colors))
        elif sys.argv[1] == "fancy_v1":
            models.append(fancy_net.Fancy_Net_v1(im_rows, im_cols, colors))
        elif sys.argv[1] == "fancy_v2":
            models.append(fancy_net.Fancy_Net_v2(im_rows, im_cols, colors))

        if sys.argv[1] != "load":
            # shuffle the training data and remove dropout proportion
            x = [x_train[j,:,:,:] for j in range(x_train.shape[0])]
            y = [y_train[j,:] for j in range(y_train.shape[0])]
            xy = zip(x, y)
            random.shuffle(xy)
            xy = xy[int(len(xy)*dropout):]
            x, y = zip(*xy)
            x = np.asarray(x)
            y = np.asarray(y)

            models[i].fit(x, y, batch_size=batch_size, \
        nb_epoch=nb_epoch, verbose=1, validation_data=(x_valid, y_valid))
            util.save_model(models[i], sys.argv[1]+'_'+str(im_rows)+'_'+str(i))

        softmax = models[i].predict(x_valid, batch_size=128, verbose=1)

        for j in range(len(y_valid)):
            predictions[j, np.argmax(softmax[j])] += 1

    top1 = 0
    for i in range(len(y_valid)):
        if np.argmax(y_valid[i]) == np.argmax(predictions[i, :]):
            top1 += 1
    top1 /= float(len(y_valid))
    print 'Final top 1 accuracy: {}, rows: {} cols: {} epoch: {}'\
            .format(top1, im_rows, im_cols, nb_epoch)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Please enter the net to train"
    else:
        main()
