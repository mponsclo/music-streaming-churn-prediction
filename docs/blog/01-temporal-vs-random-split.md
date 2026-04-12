---
title: "Temporal vs Random Split: Why 0.993 ROC-AUC Meant Nothing"
date: 2026-04-12
tags: [machine-learning, evaluation, churn-prediction, methodology]
---

# Temporal vs Random Split: Why 0.993 ROC-AUC Meant Nothing

The first LightGBM run on the KKBox churn dataset landed at ROC-AUC 0.993 with log loss 0.073 on a random 80/20 split. That is the number a junior data scientist would screenshot and put at the top of their resume. It is also the number a senior data scientist would look at and immediately distrust.

Rerunning the exact same model with a temporal holdout, train on February 2017 and test on March 2017, dropped the ROC-AUC to 0.924 and pushed the log loss to 0.267. Same features, same hyperparameters, same code path. A 7-point ROC-AUC gap, and a 266% increase in log loss.

This post is about why the second number is the honest one, and what the difference actually tells you about the model.

## What a random split secretly assumes

When you call `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` you are not sampling the future. You are sampling the same month, the same distribution, the same macro-conditions of user behavior, and then pretending your hold-out rows are unseen.

In a stationary domain that assumption is fine. Predicting digit classification on MNIST? The distribution of handwritten threes in 2026 is identical to the distribution in 2024. Your random split is a perfectly reasonable approximation of deployment.

Churn prediction is not stationary. The data has three layers of temporal structure that a random split flattens:

1. **Product-level drift**: KKBox ran different promotions, pricing, and onboarding flows between February and March 2017. The underlying user acquisition and retention dynamics shifted.
2. **Seasonal drift**: March contains the end of student subscription promotions. User mix on the platform is different.
3. **Cohort drift**: The March 2017 expiry cohort contains users whose February attempts to cancel resolved during the observation window. That resolution information is implicit in the Round 2 feature values but not in Round 1 features.

A random split mixes all three. A temporal holdout preserves them.

## The numbers

```
                    ROC-AUC   Log Loss   PR-AUC    Churn rate
Random 80/20         0.993     0.073      0.947    9.0% (test)
Temporal (Feb->Mar)  0.924     0.267      0.604    6.4% train, 9.0% test
```

![ROC and PR curves](../../outputs/figures/11_roc_pr_curves.png)

Two things jump out.

First, ROC-AUC degrades by 0.069. That is meaningful but not catastrophic. The model's *ranking ability*, its capacity to put churners above non-churners when sorted by predicted probability, survives the temporal boundary reasonably well.

Second, log loss degrades by 266%. That is catastrophic. Log loss is sensitive to probability *calibration*: how far the predicted probabilities are from the true long-run frequencies. When the churn rate shifts from 6.4% in training to 9.0% in test, the model's probability estimates become systematically biased low. It still ranks the right users at the top, but the absolute probabilities are wrong.

## Discrimination vs calibration: why the two metrics diverge

This is the subtle insight. Discrimination (can the model separate classes) and calibration (are the predicted probabilities accurate in absolute terms) are different properties.

- A model with perfect discrimination and wrong calibration would say "user A is more likely to churn than user B" correctly every time, but would report both probabilities off by a constant factor.
- A model with perfect calibration and poor discrimination would report the population-average churn rate for everyone and be locally accurate but operationally useless.

Random splits reward both properties. The training set and test set share the same base rate, so a well-fit model's calibration holds up for free. Temporal holdouts reward only discrimination. The test-set base rate has shifted, so any model trained on the old rate will be miscalibrated.

This has operational consequences. If you are running a retention campaign that says "contact users with predicted churn probability > 0.3":

- Under random-split evaluation you would contact about 12% of users
- Under temporal holdout evaluation you would contact about 17% of users
- The difference is real budget

Recalibrating the model post-deployment (Platt scaling, isotonic regression) is a standard fix, but only if you have recent labeled data. In practice you usually do not.

## Why I am glad ROC-AUC only dropped by 7 points

Look again at that 0.924. In isolation it is a strong number. In context, it tells me the model is picking up generalizable churn signal rather than temporal artifacts.

To see why, compare it to the behavioral-only ablation. When I strip out the transaction history and cancel flags and train on demographics plus listening behavior only, the random-split ROC-AUC drops to 0.771. That is the real ceiling of "predict churn before the user signals intent."

So the stack is:

- **0.993 random split full features**: ceiling under no drift. Not achievable in production.
- **0.924 temporal holdout full features**: realistic production number. What a deployed model would see.
- **0.771 random split behavioral only**: upper bound for *early* detection (before cancel signals arrive).

The distance from 0.924 to 0.771 is the value added by transaction-history and cancel-state features. The distance from 0.924 to 0.993 is the illusion of stationarity. Both matter, and you cannot see either without running both experiments.

## The methodology rule

For any prediction problem where training and serving happen at different points in time:

1. **Canonical metric comes from a temporal holdout.** That is the deployment number.
2. **Random split is an upper bound under a no-drift assumption.** Report it for completeness and to measure how much of the headline number is artifact.
3. **Monitor calibration separately from discrimination.** ROC-AUC can lie about deployment-readiness if probabilities are miscalibrated.
4. **If you see a large gap between random-split log loss and temporal log loss, expect production to need recalibration.** Plan for either periodic Platt scaling or a threshold-sweeping policy that is robust to base-rate shift.

## The interview conversation

When a hiring manager asks "how did your model do?" the junior answer is "ROC-AUC 0.993." The senior answer is "0.924 on temporal holdout, which is the honest number. 0.993 on random split, which is the upper bound under no drift. The gap tells you how much the easy evaluation was lying." That gap is where the real technical understanding lives.

## See also

- Code: [src/temporal_eval.py](../../src/temporal_eval.py) builds Round 1 features from the original KKBox files and tests against the Round 2 feature table.
- Evaluation details: [4. Evaluation](../4-evaluation.md)
- Experiment log: [5. Experiments](../5-experiments.md)
