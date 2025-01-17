import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import numpy as np

class TextEmbClassifier(torch.nn.Module):
    """
    Classifier that takes text embeddings as an input, and outputs a symptom logit

    n_emb: dimension of text embedding input
    hidden_dim: list of dimensions of hidden layers. if empty, no transformation is applied. if len>0, final dimension should be 1
    dropout_prob: dropout probability to be applied before every hidden layer.
    seed: initialization seed
    """
     
    def __init__(self, n_emb, hidden_dim, dropout_prob, seed): 

        super(TextEmbClassifier, self).__init__()

        torch.manual_seed(seed)

        self.n_emb = n_emb # embedding size of text 
        self.hidden_dim = hidden_dim # hidden dimension, if len == 0 then no transformation is applied
                                     # if len != 0, final dimension should be 1 to allow for classification

        # initialize parameters
        if len(hidden_dim) == 0: 
            self.linears = torch.nn.ModuleList([])
        else:  
            self.linears = torch.nn.ModuleList([torch.nn.Linear(self.n_emb, hidden_dim[0])])
            self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=dropout_prob)])
            prev_dim = hidden_dim[0]
            for dim in hidden_dim[1:]: 
                layer = torch.nn.Linear(prev_dim, dim)
                dropout = torch.nn.Dropout(p=dropout_prob)
                self.linears.append(layer)
                self.dropouts.append(dropout)
                prev_dim = dim
            self.hidden_activation = torch.nn.ReLU() # ReLU is used as activation after every hidden layer

    def forward(self, emb): 
        """
        forward function. transforms embedding of dim n_emb to output of size 1 (or 3) by applying linear layers
        """

        if len(self.hidden_dim) == 0: 
            return emb
        else: 
            out = emb
            for i, layer in enumerate(self.linears[:-1]): 
                out = self.dropouts[i](out)
                out = layer(out)
                out = self.hidden_activation(out) # ReLU for activation between hidden layers
            out = self.dropouts[-1](out) # if only one layer, dropout should be applied to inputs
            out = self.linears[-1](out)

        return out            


class TabularTextDataset(Dataset):
    """
    Creates a PyTorch Dataset that containing both the tabular feature vector, and the text embedding vector, as well as the symptom label
    df: the original dataframe containing the patient features (symptoms, tabular features, text notes)
    sympt: the symptom label we are trying to predict from the text and tabular features
    device: what device to put the tensors on (CPU or GPU)
    type: embedding type to use (hist, phys, both_mean, both_concat, or span)
    compl: note complexity (normal or adv)
    setting: evidence setting, i.e. what tabular features to include at the input of the classifier (all, no_sympt, realistic)
    encoder: OneHotEncoder
             if None, the encoder is fit on the training data in df 
             if not None, encoder is assumed to have been trained on training data and should now be applied as-is to test data
    scaler: StandardScaler to use for normalization of days_at_home feature 
             if None, the scaler is fit on the training data in df 
             if not None, scaler is assumed to have been trained on training data and should now be applied as-is to test data
    """
    def __init__(self, df, sympt, device, type="both_mean", compl="normal", setting="all", encoder=None, scaler=None):
        self.type = type
        self.sympt = sympt
        self.df = df.copy()
        self.device = device
        self.df[self.sympt] = self.df[self.sympt].replace({"yes":1, "no": 0, "none":0, "low": 1, "high": 2})
        self.compl = compl

        # feature selection
        features_realistic = ["asthma", "smoking", "COPD", "winter", "hay_fever", "pneu", "inf", "antibiotics"]
        symptoms = ["dysp", "cough", "pain", "fever", "nasal"]
        if setting == "all": 
            features = ["policy", "self_empl", "days_at_home"]+features_realistic+[s for s in symptoms if s != sympt]
        elif setting == "no_sympt": 
            features = ["policy", "self_empl", "days_at_home"]+features_realistic
        elif setting == "realistic": 
            features = features_realistic
        self.features = features
        self.setting = setting

        # one-hot encoding
        X_cat = df[[feat for feat in self.features if feat != "days_at_home"]]
        if encoder is None:
            self.enc = OneHotEncoder(drop='if_binary', handle_unknown='ignore')
            self.enc.fit(X_cat)
        else: # encoder was fit on training data, now applied to test data
            self.enc = encoder
        self.X_cat = self.enc.transform(X_cat).toarray() # one-hot encoded version of categorical features

        # scaling of days at home feature
        if scaler is None: 
            self.scaler = StandardScaler()
            self.scaler.fit(df[["days_at_home"]])
        else: # scaler was fit on training data, now applied to test data
            self.scaler = scaler 
        self.X_days = self.scaler.transform(df[["days_at_home"]]) # standard scaled version of days at home feature

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        returns a dictionary with the following components: 
        - sympt: contains a tensor with the symptom values for sympt
        - tab: tabular feature vector containing all the one-hot encoded categorical features, as well as the normalized days_at_home feature
        - emb: contains the text embedding for the note, constructed using one of the four strategies encoded in self.type
        """

        x = {}

        # target symptom
        sympt = self.df.iloc[idx][self.sympt]
        x[self.sympt] = torch.tensor(sympt, dtype=torch.float32, device=self.device)

        # other features
        x_cat = self.X_cat[idx]
        x_days = self.X_days[idx][0]
        if self.setting != "realistic": 
            x["tab"] = torch.tensor(np.concatenate((x_cat, [x_days])), dtype=torch.float32, device=self.device)
        else: 
            x["tab"] = torch.tensor(x_cat, dtype=torch.float32, device=self.device)

        # complexity of text 
        if self.compl == "normal": 
            hist = "hist_emb"
            phys = "phys_emb"
        elif self.compl == "adv": 
            hist = "adv_hist_emb"
            phys = "adv_phys_emb"
        else: 
            print("invalid complexity")

        # text embeddings
        if self.type == "hist": 
            x["emb"] = torch.tensor(self.df.iloc[idx][hist], device=self.device)
        elif self.type == "phys": 
            x["emb"] = torch.tensor(self.df.iloc[idx][phys], device=self.device)
        elif self.type == "both_mean": 
            emb_hist = self.df.iloc[idx][hist]
            emb_phys = self.df.iloc[idx][phys]
            emb = (emb_hist+emb_phys)/2 # total embedding is the mean of history and phys exam embedding
            x["emb"] = torch.tensor(emb, device=self.device)
        elif self.type == "both_concat": 
            emb_hist = self.df.iloc[idx][hist]
            emb_phys = self.df.iloc[idx][phys]
            emb = np.concatenate([emb_hist,emb_phys]) # total embedding is the concatenation of history and phys exam embedding
            x["emb"] = torch.tensor(emb, device=self.device)
        elif self.type == "span": 
            emb = self.df.iloc[idx][f"{self.compl}_span_{self.sympt}"] # select span embedding for this symptom
            x["emb"] = torch.tensor(emb, device=self.device, dtype=torch.float32)
        else: 
            print("invalid type")

        return x


def train_sympt_classifier(train, test, sympt, n_emb, hidden_dim, dropout, device, bs_train=100, epochs=100, seed=2023, lr=0.0001, weight_decay=1e-5, with_tab=False):
    """Train symptom classifier

    train: training data, EmbeddingDataset object
    test: validation data, EmbeddingDataset object
    sympt: name of symptom we want to predict
    n_emb: dimension of embedding 
    hidden_dim: list of dimensions to use for the hidden layers of the classifier
    dropout: dropout probability, used between each hidden layer of the classifier
    device: CPU or GPU device on which the tensors are loaded 
    bs_train: batch size 
    epochs: number of epochs
    seed: random seed
    lr: learning rate
    weight_decay: L2 regularization level
    with_tab: whether to include tabular features at input 

    returns
    - train_loss: cross-entropy score over train set for every epoch
    - test_loss: cross-entropy score over validation set for every epoch
    - model: trained model
    """
    
    torch.manual_seed(seed)
    
    train_loader = DataLoader(train, batch_size=bs_train, shuffle=True)
    if test is not None: 
        test_loader = DataLoader(test, batch_size=len(test), shuffle=False)

    # put model on the device
    model = TextEmbClassifier(n_emb, hidden_dim, dropout, seed)
    model.to(device)

    adam = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    if sympt == "fever": 
        loss = torch.nn.CrossEntropyLoss(reduction="none")
    else: 
        loss = torch.nn.BCEWithLogitsLoss(reduction="none") # all symptoms are binary, except for fever 

    train_loss = []
    test_loss = []

    for epoch in range(epochs):

        epoch_loss = 0

        for i, x in enumerate(train_loader): 

            model.train() # put model in train mode
            adam.zero_grad()

            if with_tab:
                input = torch.cat((x["tab"], x["emb"]), dim=1) # concatenate tabular features and text
            else: 
                input = x["emb"]

            if sympt == "fever": 
                logit = model(input) # predictions of model, shape (bs, 3)
                batch_loss = loss(logit, x[sympt].long()).sum()
            else: 
                logit = model(input).squeeze() # predictions of model, shape (bs,)
                batch_loss = loss(logit, x[sympt]).sum()
            
            batch_loss.backward()

            epoch_loss += batch_loss.item()
            
            # torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            adam.step()
        
        train_loss.append(epoch_loss/len(train))

        if test is not None:
            model.eval() # put model in eval mode
            with torch.no_grad():
                for x_test in test_loader: 

                    if with_tab:
                        input = torch.cat((x_test["tab"], x_test["emb"]), dim=1) # concatenate tabular features and text
                    else: 
                        input = x_test["emb"]

                    if sympt == "fever": 
                        logit = model(input) # predictions of model, shape (bs, 3)
                        batch_loss = loss(logit, x_test[sympt].long()).sum()
                    else: 
                        logit = model(input).squeeze() # predictions of model, shape (bs,)
                        batch_loss = loss(logit, x_test[sympt]).sum()
                    test_loss.append(batch_loss.item()/len(test))

    # return train_loss, test_loss, model
    return train_loss, test_loss, model


def eval_symptom_classifier(sympt, dataset, model, with_tab=False):
    """
    Evaluate trained symptom classifier
    sympt: name of symptom we are trying to predict
    dataset: EmbeddingDataset or TabularTextDataset object containing the patients to evaluate the model for
    model: trained TextEmbClassifier model 
    with_tab: whether to include tabular feature vectors at the input

    returns: precision, recall, f1, acc scores over dataset
    """

    val_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for x_val in val_loader:
            if with_tab:
                input = torch.cat((x_val["tab"], x_val["emb"]), dim=1) # concatenate tabular features and text
            else: 
                input = x_val["emb"]
            logits = model(input).squeeze()
            if sympt == "fever": 
                preds = torch.nn.functional.softmax(logits, dim=1)
                y_pred.append(torch.argmax(preds,dim=1).cpu().numpy()) # Get class with the highest probability (argmax)
            else: 
                preds = logits.sigmoid()
                y_pred.append(preds.round().cpu().numpy())  # Round predictions for binary classification -> if > 0.5, then predict class 1
            y_true.append(x_val[sympt].cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    if sympt == "fever": 
        avg_method = "macro"
    else:
        avg_method = "binary"
    precision = precision_score(y_true, y_pred, average=avg_method)
    recall = recall_score(y_true, y_pred, average=avg_method)
    f1 = f1_score(y_true, y_pred, average=avg_method)
    acc = accuracy_score(y_true, y_pred)
    
    return precision, recall, f1, acc