# SynSUM -- Synthetic Benchmark with Structured and Unstructured Medical Records

We present the SynSUM benchmark, a synthetic dataset linking unstructured clinical notes to structured background variables. The dataset consists of 10,000 artificial patient records containing tabular variables (like symptoms, diagnoses and underlying conditions) and related notes describing the fictional patient encounter in the domain of respiratory diseases. 

Should you use this dataset, please cite the [paper](https://arxiv.org/abs/2409.08936) as follows: 
```
@misc{SynSUM,
  author="Rabaey, Paloma
  and Arno, Henri
  and Heytens, Stefan
  and Demeester, Thomas",
  title="SynSUM -- Synthetic Benchmark with Structured and Unstructured Medical Records",
  year="2024",
  publisher="arXiv",
  url = {https://arxiv.org/abs/2409.08936}
}

```

## Summary

**Data access** To access the SynSUM dataset, please download the [csv file](https://github.com/prabaey/SynSUM/blob/main/SynSUM.csv) from this repository. The dataset contains the following variables (corresponding column names are between brackets): 
- diagnoses: pneumonia (`pneu`) and common cold (`cold`)
- symptoms: dyspnea (`dysp`), cough (`cough`), pain (`pain`), fever (`fever`), nasal (`nasal`)
- underlying conditions: asthma (`asthma`), smoking (`smoking`), COPD (`COPD`), hay fever (`hay_fever`)
- external influence (non-clinical): policy (`policy`), self-employed (`self_empl`), season (`season`)
- treatment: antibiotics (`antibiotics`)
- outcome: days at home (`days_at_home`)
- text note (`text`): clinical note describing the patient encounter
- compact text note (`advanced_text`): more compact (and therefore more difficult) version of the note in `text`

**Potential use** The SynSUM dataset is primarily designed to facilitate research on clinical information extraction in the presence of tabular background variables, which can be linked through domain knowledge to concepts of interest to be extracted from the text - the symptoms, in the case of SynSUM. Secondary uses include research on the automation of clinical reasoning over both tabular data and text, causal effect estimation in the presence of tabular and/or textual confounders, and multi-modal synthetic data generation.

**Data generating process** The figure below describes the full data generating process. First, the tabular portion of the synthetic patient record is sampled from a Bayesian network, where both the structure and the conditional probability distributions were defined by an expert. Afterwards, we construct a prompt containing information on the symptoms experienced by the patient, as well as their underlying health conditions (but no diagnoses). We ask the GPT-4o large language model to generate a fictional clinical note describing this patient encounter. Finally, we ask to generate a more challenging compact version of the note, mimicking the complexity of real clinical notes by prompting the use of abbreviations and shortcuts. We generate 10.000 of these synthetic patient records in total. For the full technical report on how the data was generated, we refer to the [paper](https://arxiv.org/abs/2409.08936). 

<p float="center">
<img src="img/data_generating_process.png" width="1000" />
</p>

## Additional files

`src` folder:
- `data_generating_process.py`: Contains the `RespiratoryData` class, in which the expert-defined Bayesian network is constructed. This defines the data generating process from which our tabular patient records were sampled.
- `text_generation.ipynb`: Demonstrates how text prompts are created based on the information in the tabular patient record. These prompts were fed to GPT-4o to generate the clinical text notes in our dataset.
- `symptom_predictor_baselines.ipynb`: Demonstrates how we ran some simple symptom predictor baselines on the SynSUM dataset. There are two tabular baselines (BN-tab and XGBoost-tab), one text-only neural classifier (neural-text) and one neural classifier that also sees the tabular features at the input (neural-text-tab).
- `expert_evaluation.ipynb`: Presents the results of our expert evaluation, where five experts rated the notes on several aspects, including consistency with the prompt and realism of the added context.

`utils` folder: 
- `prompt_generation.py`: Helper functions for generating the text prompts from the tabular patient records.
- `bayesian_network.py`: Helper functions for learning the Bayesian network parameters from the data, used for training the BN-tab baseline.
- `neural_classifier.py`: Helper functions for learning the neural classifier weights, used for training the neural-text and neural-text-tab baselines. 
