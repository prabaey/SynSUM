import torch
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import itertools

class CustomDataset(Dataset):
    """
    Simple dataset used for MLE training loop
    df: dataframe containing the tabular features
    """
    def __init__(self, df):
        self.df = df.replace({"yes":1, "no": 0, "none":0, "low": 1, "high": 2}) # text feature categories are transformed into numerical categories

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # each item is a dictionary with as keys the names of all tabular variables, followed by their value in tensor format
        x = {}
        for col in self.df.columns:
            x[col] = torch.tensor(self.df.iloc[idx][col], dtype=torch.float32)
        return x

class NoisyOr(torch.nn.Module):
    """
    Noisy OR distribution 
    outcome: name of symptom we are modeling
    parents: list of parent variables 
    """
    
    def __init__(self, outcome, parents): 
        super(NoisyOr, self).__init__()
        self.outcome = outcome # name of variable we are modeling
        self.parents = parents # parent variables
        self.n = len(parents) # number of parents

        # learnable parameters
        self.lambda_0 = torch.nn.Parameter(torch.rand(1)) # leak probability
        self.lambdas = torch.nn.Parameter(torch.rand(self.n)) # activation probabilities
        
    def forward(self, sample): 
        """
        calculate the log-probability that a symptom is activated/not activated (depending on its observed value), given its parent values 
        sample: datapoint (dict) containing symptom and parent values as tensors
        """

        lambda_0 = torch.sigmoid(self.lambda_0) # constrain between 0 and 1 
        lambdas = torch.sigmoid(self.lambdas) # constrain between 0 and 1
        
        y = sample[self.outcome] # select the outcome (symptom that is activated)
        x = torch.stack([sample[parent] for parent in self.parents], dim=1) # select the parents and stack them into a tensor of dim (bs, n)
        
        prod = (1-lambda_0)*torch.prod((1-lambdas)**x, dim=1) # probability that symptom is not active 
        
        log_p = torch.where(y==1, torch.log(1-prod), torch.log(prod)) # probability that symptom is active vs. not active 
        
        return log_p

    def train(self, df, bs=50, lr=0.01, num_epochs=10):
        """
        Estimate parameters of the noisyOR distribution (lambda_0 and lambdas) from dataset 
        df: dataset containing all tabular variables (dict with variable name as key, and tensor as value)
        bs: batch size
        lr: learning rate
        num_epochs: number of epochs
        """

        df_subset = df[[self.outcome]+self.parents]
        train_data = CustomDataset(df_subset)
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)

        optimizer = Adam(self.parameters(), lr=lr) 

        for _ in range(num_epochs):
            for batch in train_loader:

                # Forward pass
                log_prob = self.forward(batch)
                loss = -log_prob.sum() # negative log-likelihood

                # Backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def get_CPT(self): 
        """
        Generate full conditional probability table from the noisy-OR parameters. 
        Contains P(outcome | parents) for all combinations of parents, where parents are assumed to have two possible values (yes/no)
        First row of table contains P(outcome = yes | parents)
        Second row of table contains P(outcome = no | parents)
        """

        combinations = itertools.product([1, 0], repeat=len(self.parents))

        input = {parent:[] for parent in self.parents}

        for comb in combinations: 
            for i, parent in enumerate(self.parents): 
                input[parent].append(comb[i])

        input[self.outcome] = torch.ones(2**self.n)
        input = {parent:torch.tensor(val) for parent, val in input.items()}

        p_pos = self.forward(input).exp() # prob(outcome | parents) for all combinations of parents
        p_neg = 1-p_pos

        cpt = torch.stack((p_pos, p_neg))

        return cpt.detach().clone().numpy()
    

class Antibiotics(torch.nn.Module):
    """
    Conditional probability distribution for Antibiotics variable, parameterized using a logistic regression model
    See also Antibiotics class in data_generating_process.py
    outcome: name of variable we are modeling (antibiotics)
    parents: list of parent variables 
    """
    
    def __init__(self, outcome, parents): 

        super(Antibiotics, self).__init__()
        self.outcome = outcome # name of variable we are modeling
        self.parents = parents # parent variables
        self.n = len(parents) # number of parents

        # learnable parameters
        self.bias = torch.nn.Parameter(torch.rand(1)) # bias
        self.coeff = torch.nn.Parameter(torch.rand(self.n+1)) # coefficients (2 for fever!)
        
    def forward(self, sample): 
        """
        calculate the log-probability that antibiotics is prescribed/not prescribed (depending on its observed value), given the values of the parent variables
        sample: datapoint (dict) containing tabular values as tensors
        """

        low_fever = torch.where(sample["fever"] == 1, torch.ones_like(sample["fever"]), torch.zeros_like(sample["fever"]))
        high_fever = torch.where(sample["fever"] == 2, torch.ones_like(sample["fever"]), torch.zeros_like(sample["fever"]))

        y = sample[self.outcome] # select the outcome (antibiotics)
        
        logit = self.bias + self.coeff[0]*sample["policy"] \
                + self.coeff[1]*sample["dysp"] + self.coeff[2]*sample["cough"] \
                + self.coeff[3]*sample["pain"] \
                + self.coeff[4]*low_fever + self.coeff[5]*high_fever
        prob = torch.sigmoid(logit)
        log_p = torch.where(y==1, torch.log(prob), torch.log(1-prob))

        return log_p
    
    def train(self, df, bs=50, lr=0.01, num_epochs=15):
        """
        Estimate parameters of the Antibiotics distribution (bias and coeff) from dataset 
        df: dataset containing all tabular variables (dict with variable name as key, and tensor as value)
        bs: batch size
        lr: learning rate
        num_epochs: number of epochs
        """

        df_subset = df[[self.outcome]+self.parents]
        train_data = CustomDataset(df_subset)
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)

        optimizer = Adam(self.parameters(), lr=lr) 

        for _ in range(num_epochs):
            for batch in train_loader:

                # Forward pass
                log_prob = self.forward(batch)
                loss = -log_prob.sum() # negative log-likelihood

                # Backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def get_CPT(self): 
        """
        Generate full conditional probability table from the regression model parameters. 
        Contains P(outcome | parents) for all combinations of parents, where parents are assumed to have two possible values (yes/no), except for fever
        First row of table contains P(outcome = yes | parents)
        Second row of table contains P(outcome = no | parents)
        """

        input = {parent:[] for parent in self.parents}

        for f in [2, 1, 0]: # possible values for fever:
            combinations = itertools.product([1, 0], repeat=len(self.parents)-1) # all parents except for fever
            for comb in combinations: 
                for i, parent in enumerate(self.parents): 
                    if parent == "fever": 
                        input[parent].append(f)
                    else: 
                        input[parent].append(comb[i])

        input[self.outcome] = torch.ones(len(input["fever"]))
        input = {parent:torch.tensor(val) for parent, val in input.items()}

        p_pos = self.forward(input).exp() # prob(outcome | parents) for all combinations of parents
        p_neg = 1-p_pos

        cpt = torch.stack((p_pos, p_neg))

        return cpt.detach().clone().numpy()
    

class DaysAtHome(torch.nn.Module):
    """
    Conditional probability distribution for days_at_home variable, parameterized using a Poisson regression model
    See also DaysAtHome class in data_generating_process.py
    outcome: name of variable we are modeling (days_at_home)
    parents: list of parent variables 
    """
    
    def __init__(self, outcome, parents): 

        super(DaysAtHome, self).__init__()
        self.outcome = outcome # name of variable we are modeling
        self.parents = parents # parent variables
        self.n = len(parents) # number of parents

        # learnable parameters antibiotics=0 model
        self.bias_a0 = torch.nn.Parameter(torch.rand(1)) # bias
        self.coeff_a0 = torch.nn.Parameter(torch.rand(self.n)) # coefficients (2 for fever but one less for antibiotics)

        # learnable parameters antibiotics=1 model
        self.bias_a1 = torch.nn.Parameter(torch.rand(1)) # bias
        self.coeff_a1 = torch.nn.Parameter(torch.rand(self.n)) # coefficients (2 for fever but one less for antibiotics)
        
    def forward(self, sample): 
        """
        calculate the log-probability that patient stays at home for a specific number of days (depending on its observed value), given the values of the parent variables
        sample: datapoint (dict) containing tabular values as tensors
        """

        low_fever = torch.where(sample["fever"] == 1, torch.ones_like(sample["fever"]), torch.zeros_like(sample["fever"]))
        high_fever = torch.where(sample["fever"] == 2, torch.ones_like(sample["fever"]), torch.zeros_like(sample["fever"]))

        y = sample[self.outcome] # select the outcome (days at home)
        
        logit_a0 = self.bias_a0 + self.coeff_a0[0]*sample["self_empl"] \
                + self.coeff_a0[1]*sample["dysp"] + self.coeff_a0[2]*sample["cough"] \
                + self.coeff_a0[3]*sample["pain"] + self.coeff_a0[4]*sample["nasal"] \
                + self.coeff_a0[5]*low_fever + self.coeff_a0[6]*high_fever
        logit_a1 = self.bias_a1 + self.coeff_a1[0]*sample["self_empl"] \
                + self.coeff_a1[1]*sample["dysp"] + self.coeff_a1[2]*sample["cough"] \
                + self.coeff_a1[3]*sample["pain"] + self.coeff_a1[4]*sample["nasal"] \
                + self.coeff_a1[5]*low_fever + self.coeff_a1[6]*high_fever
        log_lambda = torch.where(sample["antibiotics"] == 1, logit_a1, logit_a0) # log(lambda), where labmda is mean of poisson distr

        log_p = y*log_lambda-log_lambda.exp()-torch.lgamma(y+1) # log of Poisson probability (k*log(lambda)-lambda-log(k!)), where k! = lgamma(k+1)
        
        return log_p
    
    def train(self, df, bs=50, lr=0.01, num_epochs=15):
        """
        Estimate parameters of the Antibiotics distribution (bias and coeff) from dataset 
        df: dataset containing all tabular variables (dict with variable name as key, and tensor as value)
        bs: batch size
        lr: learning rate
        num_epochs: number of epochs
        """

        df_subset = df[[self.outcome]+self.parents]
        train_data = CustomDataset(df_subset)
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)

        optimizer = Adam(self.parameters(), lr=lr) 

        for _ in range(num_epochs):
            for batch in train_loader:

                # Forward pass
                log_prob = self.forward(batch)
                loss = -log_prob.sum() # negative log-likelihood

                # Backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def get_CPT(self):
        """
        Generate full conditional probability table from the Poisson model parameters. 
        Contains P(outcome | parents) for all combinations of parents, where parents are assumed to have two possible values (yes/no), except for fever
        First row of table contains P(days_at_home = 0 | parents)
        Second row of table contains P(days_at_home = 1 | parents)
        etc. 
        Last row of table contains P(days_at_home >= 15 | parents) (15 is the maximum number of days observed in the training data)
        """

        input = {parent:[] for parent in self.parents}

        for f in [2, 1, 0]: # possible values for fever:
            combinations = itertools.product([1, 0], repeat=len(self.parents)-1) # all parents except for fever
            for comb in combinations: 
                for i, parent in enumerate(self.parents): 
                    if parent == "fever": 
                        input[parent].append(f)
                    else: 
                        input[parent].append(comb[i])

        cpt = torch.empty((0, len(input["fever"])), dtype=torch.float32)
        for days in range(15): # days range from 0 to 14
            input[self.outcome] = torch.ones(len(input["fever"]))*days
            input = {parent:torch.tensor(val) for parent, val in input.items()}
            p_pos = self.forward(input).exp() # prob(outcome | parents) for all combinations of parents
            cpt = torch.cat((cpt, p_pos.unsqueeze(dim=0)), dim=0)

        p_rest = 1 - torch.sum(cpt, dim=0) # P(k > 14) = 1-P(k<=14) = 1-P(k=0)-P(k=1)-...-P(k=13)-P(k=14)

        cpt = torch.cat((cpt, p_rest.unsqueeze(dim=0)), dim=0)

        return cpt.detach().clone().numpy()