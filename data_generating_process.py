from scipy.stats import poisson
import numpy as np

class CPD: 
    """
    Conditional Probability Distribution
    
    var_name: name of the variable
    cpd: conditional probability table, using ordering of parents and their values in self.parent_value_order
    parents: list of parent names in order followed by cpd
    value_order: ordering of values of the variable var_name, which is followed by the rows of the CPD
    parent_value_order: dict with ordering of values per parent, which is followed by the columns of the CPD
    """
    # only consider binary parents
    def __init__(self, var_name, cpd, parents, value_order, parent_value_order): 
        
        self.var_name = var_name # name of variable 
        self.parents = parents # list of parent names (ordering will be used)
        self.table = cpd # conditional probability table, using order or parents and their values in self.parent_value_order
        self.parent_value_order = parent_value_order # dict with order of values per parent
        self.value_order = value_order # variable value represented by each row of the CPD, in order
        
    def get_prob(self, evidence): 
        # evidence is dict {parent: value}
        # return dict with probability for each value
        
        idxs = [self.parent_value_order[parent].index(evidence[parent]) for parent in evidence]
            
        n = len(self.parents)
        total_idx = 0
        for i, idx in enumerate(idxs): 
            shift = 2**(n-i-1)
            total_idx += idx*shift
            
        prob = self.table[:, total_idx]
        return {self.value_order[i]:prob[i] for i in range(len(prob))}  
    
class noisy_OR: 
    """
    Noisy OR distribution
    var_name: name of the variable
    parents: list of parent names, provides ordering used in success_prob
    leak_prob: leak probability
    success_prob: list of success probabilities, using ordering of parents
    """
    def __init__(self, var_name, parents, leak_prob, success_prob): 
        self.var_name = var_name # binary variable (yes/no)
        self.parents = parents # ordering of parents, they are always binary 
        self.leak_prob = leak_prob # leak probability 
        self.success_prob = success_prob # list of success probabilities, uses ordering of self.parents
        
    def get_prob(self, activated_parents):
        # list of activated_parents contains parents which are "on"
        # returns success probability for binary variable (P(var_name=yes|...))
        total_p = (1-self.leak_prob) # (1-p0)
        for parent in activated_parents: 
            idx = self.parents.index(parent)
            factor = (1-self.success_prob[idx]) # (1-pi) for all i which are activated -> prob. that activation fails
            total_p *= factor
        
        return 1-total_p # P(var_name=yes|...) = 1-P(var_name=no)

class Antibiotics:
    """
    Logistic regression model for Antibiotics
    coeff: list of coefficients with order [bias, policy, dysp, cough, pain, low_fever, high_fever]
    """
    def __init__(self, coeff): 
        self.coeff = coeff
        
    def __call__(self, policy, dysp, cough, pain, fever): 
        
        policy = 0 if policy == "no" else 1
        dysp = 0 if dysp == "no" else 1
        cough = 0 if cough == "no" else 1
        pain = 0 if pain == "no" else 1
        
        low_fever = 0
        high_fever = 0
        if fever == "low": 
            low_fever = 1
        if fever == "high": 
            high_fever = 1
            
        logit = self.coeff[0] + self.coeff[1]*policy \
                + self.coeff[2]*dysp + self.coeff[3]*cough \
                + self.coeff[4]*pain \
                + self.coeff[5]*low_fever + self.coeff[6]*high_fever
        prob = 1 / (1 + np.exp(-logit))
        
        return prob

class DaysAtHome:
    """
    Poisson regression model for DaysAtHome
    coeff: list of coefficients with order [dysp, cough, pain, nasal, low_fever, high_fever, self_employed]
    bias: bias
    """
    
    def __init__(self, coeff, bias): 
        self.bias = bias
        self.coeff = coeff
        
    def get_mean(self, dysp, cough, pain, nasal, fever, self_employed): 
        
        dysp = 0 if dysp == "no" else 1
        cough = 0 if cough == "no" else 1
        pain = 0 if pain == "no" else 1
        nasal = 0 if nasal == "no" else 1
        self_employed = 0 if self_employed == "no" else 1
        
        low_fever = 0
        high_fever = 0
        if fever == "low": 
            low_fever = 1
        if fever == "high": 
            high_fever = 1
        
        logit = self.bias \
                + self.coeff[0]*dysp + self.coeff[1]*cough \
                + self.coeff[2]*pain + self.coeff[3]*nasal \
                + self.coeff[4]*low_fever + self.coeff[5]*high_fever \
                + self.coeff[6]*self_employed
        
        lambda_ = np.exp(logit)
        
        return lambda_


class RespiratoryData:
    """
    Class defining the full joint probability distribution for our tabular patient records. 
    seed: seed used for sampling
    generate_sample: generates a synthetic patient according to the joint distribution, using top-down sampling
    """
    
    def __init__(self, seed): 
        
        # set seed
        np.random.seed(seed)
        
        # root nodes
        self.p_policy = 0.65
        self.p_self_empl = 0.11
        self.p_astma = 0.095
        self.p_smoking = 0.19
        self.p_hay_fever = 0.015
        self.p_winter = 0.4
        
        # diagnoses
        table_pneu = np.array([[0.995, 0.985, 0.9935, 0.98, 0.987, 0.96, 0.987, 0.96], #(COPD=0, astma=0, winter=0), (COPD=0, astma=0, winter=1), (COPD=0, astma=1, winter=0), (COPD=0, astma=1, winter=1), (COPD=1, astma=0, winter=0), (COPD=1, astma=0, winter=1), (COPD=1, astma=1, winter=0), (COPD=1, astma=1, winter=1)
                               [0.005, 0.015, 0.0065, 0.02, 0.013, 0.04, 0.013, 0.04]]) 
        self.cpd_pneu = CPD("pneu", table_pneu, ["COPD", "astma", "winter"], ["no", "yes"], {"COPD": ["no", "yes"], "astma": ["no", "yes"], "winter": ["no", "yes"]}) 
        
        table_inf = np.array([[0.95, 0.5], #(winter=no), (winter=yes)
                              [0.05, 0.5]])
        self.cpd_inf = CPD("inf", table_inf, ["winter"], ["no", "yes"], {"winter": ["no", "yes"]})
        
        table_COPD = np.array([[0.9925, 0.927], #(smoking=no), (smoking=yes)
                               [0.0075, 0.073]])
        self.cpd_COPD = CPD("COPD", table_COPD, ["smoking"], ["no", "yes"], {"smoking": ["no", "yes"]})
        
        # symptoms
        self.dysp_noisy_OR = noisy_OR("dysp", ["astma", "smoking", "COPD", "hay_fever", "pneu"], 0.05, [0.9, 0.3, 0.9, 0.2, 0.3])
        self.cough_noisy_OR = noisy_OR("cough", ["astma", "smoking", "COPD", "pneu", "inf"], 0.07, [0.3, 0.6, 0.4, 0.85, 0.7])
        self.pain_noisy_OR = noisy_OR("pain", ["COPD", "cough", "pneu", "inf"], 0.05, [0.15, 0.2, 0.3, 0.1])
        self.nasal_noisy_OR = noisy_OR("nasal", ["hay_fever", "inf"], 0.1, [0.85, 0.7])
        table_fever = np.array([[0.80, 0.75, 0.10, 0.05], #(pneu=no,inf=no), (pneu=no,inf=yes), (pneu=yes, inf=no), (pneu=yes, inf=yes)
                                [0.15, 0.20, 0.10, 0.15],
                                [0.05, 0.05, 0.80, 0.80]])
        self.cpd_fever = CPD("fever", table_fever, ["pneu", "inf"], ["none", "low", "high"], {"pneu": ["no", "yes"], "inf": ["no", "yes"]})
        
        # exposure
        self.model_antibiotics = Antibiotics([-3, 2/2, 1.6/2, 1.33/2, 1.33/2, 1.8/2, 4.5/2]) # bias, policy, dysp, cough, pain, fever_low, fever_high
        
        # outcome
        self.model_days_no_antibiotics = DaysAtHome([0.64, 0.35, 0.47, 0.011, 0.81, 1.23, -0.5], 0.010) # dysp, cough, pain, nasal, fever_low, fever_high, self_empl
        self.model_days_antibiotics = DaysAtHome([0.51, 0.42, 0.26, 0.0051, 0.24, 0.57, -0.5], 0.16) # dysp, cough, pain, nasal, fever_low, fever_high, self_empl
        
    def sample_from_cpd(self, cpd, evidence): 
        # helper function for sampling from CPD class
        # cpd: conditional probability distribution to sample from (CPD Class instance)
        # evidence: dict {parent: value} containing parent values to condition on
            
        probs = cpd.get_prob(evidence)
        out_values = cpd.value_order
        p = [probs[val] for val in out_values]
        sample = np.random.choice(out_values, 1, p=p)[0]
        return sample
        
    def generate_sample(self): 

        # root variables
        policy = np.random.choice(["no", "yes"], 1, p=[1-self.p_policy, self.p_policy])[0]
        self_empl = np.random.choice(["no", "yes"], 1, p=[1-self.p_self_empl, self.p_self_empl])[0]
        astma = np.random.choice(["no", "yes"], 1, p=[1-self.p_astma, self.p_astma])[0]
        smoking = np.random.choice(["no", "yes"], 1, p=[1-self.p_smoking, self.p_smoking])[0]
        hay_fever = np.random.choice(["no", "yes"], 1, p=[1-self.p_hay_fever, self.p_hay_fever])[0]
        winter = np.random.choice(["no", "yes"], 1, p=[1-self.p_winter, self.p_winter])[0]

        # diagnoses
        COPD = self.sample_from_cpd(self.cpd_COPD, {"smoking": smoking})
        pneu = self.sample_from_cpd(self.cpd_pneu, {"COPD": COPD, "astma": astma, "winter": winter})
        inf = self.sample_from_cpd(self.cpd_inf, {"winter": winter}) # note: "cold" was renamed "inf" (infection of the upper respiratory airways)
                                                                     # to avoid confusion with "winter"
        
        # symptoms 
        background_nodes = {"astma": astma, "smoking": smoking, "hay_fever": hay_fever, "winter": winter,
                           "COPD": COPD, "pneu": pneu, "inf": inf}
        
        p_dysp = self.dysp_noisy_OR.get_prob([node for node in ["astma", "smoking", "COPD", "hay_fever", "pneu"] if background_nodes[node] == "yes"])
        dysp = np.random.choice(["no", "yes"], 1, p=[1-p_dysp, p_dysp])[0]
        
        p_cough = self.cough_noisy_OR.get_prob([node for node in ["astma", "smoking", "COPD", "pneu", "inf"] if background_nodes[node] == "yes"])
        cough = np.random.choice(["no", "yes"], 1, p=[1-p_cough, p_cough])[0]
        background_nodes["cough"] = cough
        
        p_pain = self.pain_noisy_OR.get_prob([node for node in ["COPD", "cough", "pneu", "inf"] if background_nodes[node] == "yes"])
        pain = np.random.choice(["no", "yes"], 1, p=[1-p_pain, p_pain])[0]
        
        p_nasal = self.nasal_noisy_OR.get_prob([node for node in ["hay_fever", "inf"] if background_nodes[node] == "yes"])
        nasal = np.random.choice(["no", "yes"], 1, p=[1-p_nasal, p_nasal])[0]
        
        fever = self.sample_from_cpd(self.cpd_fever, {"pneu": pneu, "inf": inf})
        
        # exposure
        p_antibio = self.model_antibiotics(policy, dysp, cough, pain, fever)
        antibiotics = np.random.choice(["no", "yes"], 1, p=[1-p_antibio, p_antibio])[0]
            
        # outcome 
        if antibiotics == "no": 
            lambda_ = self.model_days_no_antibiotics.get_mean(dysp, cough, pain, nasal, fever, self_empl)
            days_at_home = poisson.rvs(lambda_, size=1)[0]
        else: 
            lambda_ = self.model_days_antibiotics.get_mean(dysp, cough, pain, nasal, fever, self_empl)
            days_at_home = poisson.rvs(lambda_, size=1)[0]
            
        return {"policy": policy, "self_empl": self_empl, "asthma": astma, "smoking": smoking, "COPD": COPD, "winter": winter,
                "hay_fever": hay_fever, "pneu": pneu, "inf": inf, "dysp": dysp, "cough": cough, "pain": pain, "fever": fever,
                "nasal": nasal, "antibiotics": antibiotics, "days_at_home": days_at_home}