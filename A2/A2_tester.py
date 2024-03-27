import A2_A0262349Y as grading
import A2_A0257926N as grading_jsim


for N in range(1, 9):
    print("N = ", N)
    X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array = grading.A2_A0262349Y(N)
    # print("error_train_array\n", error_train_array)
    # print("error_test_array\n", error_test_array)
    
    print("type(X_train): ", type(X_train))
    print("type(y_train): ", type(y_train))
    print("type(X_test): ", type(X_test))
    print("type(y_test): ", type(y_test))
    print("type(Ytr): ", type(Ytr))
    print("type(Yts): ", type(Yts))
    print("type(Ptrain_list): ", type(Ptrain_list))
    print("type(Ptest_list): ", type(Ptest_list))
    print("type(w_list): ", type(w_list))
    print("type(error_train_array): ", type(error_train_array))
    print("type(error_test_array): ", type(error_test_array))
    
    
    
for N in range(1, 9):
    print("N = ", N)
    X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array = grading_jsim.A2_A0257926N(N)
    # print("error_train_array\n", error_train_array)
    # print("error_test_array\n", error_test_array)
    
    print("type(X_train): ", type(X_train))
    print("type(y_train): ", type(y_train))
    print("type(X_test): ", type(X_test))
    print("type(y_test): ", type(y_test))
    print("type(Ytr): ", type(Ytr))
    print("type(Yts): ", type(Yts))
    print("type(Ptrain_list): ", type(Ptrain_list))
    print("type(Ptest_list): ", type(Ptest_list))
    print("type(w_list): ", type(w_list))
    print("type(error_train_array): ", type(error_train_array))
    print("type(error_test_array): ", type(error_test_array))