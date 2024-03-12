import A2_A0262349Y as grading

for N in range(1, 9):
    print("N = ", N)
    X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array = grading.A2_A0262349Y(N)
    print("error_train_array\n", error_train_array)
    print("error_test_array\n", error_test_array)