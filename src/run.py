import alex_net
import inception_v3_lowres as v3
import util
import sys

def main():
    img_rows, img_cols = 79, 79
    batch_size         = 16
    nb_epoch           = 2
    random_state       = 51
    colors             = 1

    train_data, train_target, driver_id, unique_drivers = util.load_train_data(img_rows, img_cols, colors)
    test_data, test_id = util.load_test_data(img_rows, img_cols, colors)

    yfull_train = dict()
    yfull_test = []
    unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                     'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                     'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                     'p075']
    x_train, y_train, train_index = util.copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)
    unique_list_valid = ['p081']
    x_valid, y_valid, test_index = util.copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)

    print 'Starting Single Run'
    print ('Split train: ', len(x_train), len(y_train))
    print ('Split valid: ', len(x_valid), len(y_valid))
    print ('Train drivers: ', unique_list_train)
    print ('Test drivers: ', unique_list_valid)

    if sys.argv[1] == "alex_v1":
        model = create_model_v1(img_rows, img_cols, colors)
    elif sys.argv[1] == "alex_v2":
        model = create_model_v2(img_rows, img_cols, colors)
    elif sys.argv[1] == "inception_v3":
        model = v3.InceptionV3(include_top=True, weights=None)

    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(x_valid, y_valid))

    score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Score log_loss: ', score[0])

    predictions_valid = model.predict(x_valid, batch_size=128, verbose=1)
    score = log_loss(y_valid, predictions_valid)
    print('Score log_loss: ', score)

    # Store valid predictions
    for i in range(len(test_index)):
        yfull_train[test_index[i]] = predictions_valid[i]

    # Store test predictions
    test_prediction = model.predict(test_data, batch_size=128, verbose=1)
    yfull_test.append(test_prediction)

    print 'Final log_loss: {}, rows: {} cols: {} epoch: {}'.format(score, img_rows, img_cols, nb_epoch)
    # info_string = 'loss_' + str(score) \
    #                 + '_r_' + str(img_rows) \
    #                 + '_c_' + str(img_cols) \
    #                 + '_ep_' + str(nb_epoch)
    #
    # test_res = merge_several_folds_mean(yfull_test, 1)
    # create_submission(test_res, test_id, info_string)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Please enter the net to train"
    else:
        main()
