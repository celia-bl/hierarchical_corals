import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin


class MLPHead(BaseEstimator, nn.Module):
    def __init__(self, hidden_layers_sizes, lr=0.001, epochs=200, batch_size=200, output_dim=None):
        super(MLPHead, self).__init__()

        self.hidden_layers_sizes = hidden_layers_sizes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"

        self.model = None  # Model will be built in the fit method
        self.input_dim = None
        self.output_dim = output_dim
        self.loss_curve_ = []
        self.is_trained = False

    def _build_model(self):
        layers = [nn.Linear(self.input_dim, self.hidden_layers_sizes[0]), nn.ReLU()]
        for i in range(1, len(self.hidden_layers_sizes)):
            layers.append(nn.Linear(self.hidden_layers_sizes[i - 1], self.hidden_layers_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_layers_sizes[-1], self.output_dim))

        self.model = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X_tensor = X.to(self.device)

        print(y)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        y_tensor = y.to(self.device)

        new_input_dim = X_tensor.shape[1]
        # nombre total de classe dans le labelset de rio do fogo = 145
        if self.output_dim is None:
            new_output_dim = len(torch.unique(y_tensor))
        else:
            new_output_dim = self.output_dim
        # print(y_tensor)
        # new_output_dim = 145
        if self.model is None or self.input_dim != new_input_dim or self.output_dim is not new_output_dim:
            self.input_dim = new_input_dim
            self.output_dim = new_output_dim
            self._build_model()
            print("Build new model")

        if not self.is_trained:
            self.model.train()
            self.is_trained = True

        #print("!!!!! PARAMS !!!!")
        #for name, param in self.model.named_parameters():
        #    print(name, param.data)
        #y_pred = self.predict(x_test)
        #accuracy = accuracy_score(y_test, y_pred)
        #print("Test accuracy model", accuracy)
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)  # inputs)
            loss = self.criterion(outputs, y_tensor)  # labels)
            loss.backward()
            self.optimizer.step()

            #y_pred = self.predict(x_test)
            #accuracy = accuracy_score(y_test, y_pred)
            #print("epochs", epoch, "test accuracy", accuracy)

            self.loss_curve_.append(loss.item())
        #print("!!!!!!!!! PARAMS 2 !!!!!!!!!!")
        #for name, param in self.model.named_parameters():
        #    print(name, param.data)
        return self

    def predict_and_proba(self, X):
        X_tensor = X

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            #print('PROBA', probabilities.shape)
            score, predicted = torch.max(probabilities, 1)
            #print('PREDICTED', predicted.shape)
            # score, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy(), score.cpu().numpy(), probabilities.cpu()

    def predict(self, X):
        X_tensor = X

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()

    def score(self, X, y):
        y_pred, _ = self.predict(self, X)
        return accuracy_score(y, y_pred)
