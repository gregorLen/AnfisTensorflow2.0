# Test myanfis model
import myanfis
import numpy as np

if __name__ == "__main__":
    # set parameters
    param = myanfis.fis_parameters(
        n_input=2,                # no. of Regressors
        n_memb=2,                 # no. of fuzzy memberships
        batch_size=16,            # 16 / 32 / 64 / ...
        memb_func='sigmoid',      # 'gaussian' / 'gbellmf' / 'sigmoid'
        optimizer='adam',          # sgd / adam / ...
        # mse / mae / huber_loss / mean_absolute_percentage_error / ...
        loss='huber_loss',
        n_epochs=50               # 10 / 25 / 50 / 100 / ...
    )
    # create random data
    X_train = np.random.rand(param.batch_size * 5, param.n_input),
    X_test = np.random.rand(param.batch_size * 2, param.n_input)
    y_train = np.random.rand(param.batch_size * 5, 1),
    y_test = np.random.rand(param.batch_size * 2, 1)

    fis = myanfis.ANFIS(n_input=param.n_input,
                        n_memb=param.n_memb,
                        batch_size=param.batch_size,
                        memb_func=param.memb_func,
                        name='myanfis'
                        )

    # compile model
    fis.model.compile(optimizer=param.optimizer,
                      loss=param.loss
                      # ,metrics=['mse']  # ['mae', 'mse']
                      )

    # fit model
    history = fis.fit(X_train, y_train,
                      epochs=param.n_epochs,
                      batch_size=param.batch_size,
                      validation_data=(X_test, y_test),
                      # callbacks = [tensorboard_callback]  # for tensorboard
                      )

    # eval model
    import pandas as pd
    fis.plotmfs(show_initial_weights=True)

    loss_curves = pd.DataFrame(history.history)
    loss_curves.plot(figsize=(8, 5))

    fis.model.summary()

    # get premise parameters
    premise_parameters = fis.model.get_layer(
        'fuzzyLayer').get_weights()       # alternative

    # get consequence paramters
    bias = fis.bias
    weights = fis.weights
    # conseq_parameters = fis.model.get_layer('defuzzLayer').get_weights()       # alternative


# manually check ANFIS Layers step-by-step

    # L1 = myanfis.FuzzyLayer(n_input, n_memb)
    # L1(X) # to call build function
    # mus = fis.mus
    # sigmas = fis.sigmas
    # L1.set_weights([fis.mus, fis.sigmas])

    # op1 = np.array(L1(Xs))

    # L2 = myanfis.RuleLayer(n_input, n_memb)
    # op2 = np.array(L2(op1))

    # L3 = myanfis.NormLayer()
    # op3 = np.array(L3(op2))

    # L4 = myanfis.DefuzzLayer(n_input, n_memb)
    # L4(op3, Xs) # to call build function
    # bias = fis.bias
    # weights = fis.weights
    # L4.set_weights([fis.bias, fis.weights])
    # op4 = np.array(L4(op3, Xs))

    # L5 = myanfis.SummationLayer()
    # op5 = np.array(L5(op4))
