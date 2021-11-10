from autoPyTorch import AutoNetClassification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

autoPyTorch = AutoNetClassification("full_cs",  # config preset
                                    log_level='info',
                                    max_runtime=300,
                                    min_budget=30,
                                    max_budget=90,
                                    random_seed=2)

autoPyTorch.fit(X_train, y_train, validation_split=0.3)
y_pred = autoPyTorch.predict(X_test)

print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))

print("======== PyTorch MODEL ========")
print(autoPyTorch.get_pytorch_model())
print("======== AutoNet CONFIG ========")
print(autoPyTorch.get_current_autonet_config())
print("======== Hyperparameter ========")
print(autoPyTorch.get_hyperparameter_search_space())
