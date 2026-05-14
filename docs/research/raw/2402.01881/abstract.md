# Large Language Model Agent for Hyper-Parameter Optimization (AgentHPO)

Source: https://arxiv.org/abs/2402.01881

## Verbatim abstract

Hyperparameter optimization is critical in modern machine learning, requiring
expert knowledge, numerous trials, and high computational and human resources.
Despite the advancements in Automated Machine Learning (AutoML), challenges in
terms of trial efficiency, setup complexity, and interoperability still
persist. To address these issues, we introduce a novel paradigm leveraging
Large Language Models (LLMs) to automate hyperparameter optimization across
diverse machine learning tasks, which is named AgentHPO (short for LLM
Agent-based Hyperparameter Optimization). Specifically, AgentHPO processes the
task information autonomously, conducts experiments with specific
hyperparameters (HPs), and iteratively optimizes them based on historical
trials. This human-like optimization process largely reduces the number of
required trials, simplifies the setup process, and enhances interpretability
and user trust, compared to traditional AutoML methods. Extensive empirical
experiments conducted on 12 representative machine-learning tasks indicate
that AgentHPO not only matches but also often surpasses the best human trials
in terms of performance while simultaneously providing explainable results.
