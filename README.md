

<h1 style="text-align:center;">Loan Approval Project</h1>

<p style="text-align:center;">By Blake Dennett<p>


<h1>Table of Contents</h1>

* <a href="#project-summary">Project Summary</a>

* <a href="#visualizations">Power Bi Visuals</a>

* <a href="#data-analysis">Data Analysis/Preprocessing</a>

* <a href="#machine-learning">Machine Learning</a>

    * <a href="#decision-tree">Decision Tree</a>
    * <a href="#neural-network">Neural Network</a>

* <a href="#acknowledgments">Acknowledgements</a>


## Project Summary
<a href="https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset">Kaggle Dataset<a>



## Visualizations


#### Loan Status Comparisons 

<img src="./images/TripleLoanStatus.PNG" alt="Income Dependent Graph">

<h6>The credit score has a huge average difference for rejected and approved loans while education, self-employment, and the term of the loan don't seem to be impactful.</h6>

<br><br>

#### Benford's Law

<h6>Benford's law deals with the first digit of each number in an exponentially increasing set of values. It states that lower numbers should occur more frequently as seen in the graph below. This law is used for fraud detection.</h6>

<img src="./images/BenfordExample.PNG">

<img src="./images/BenfordActual.PNG">

##### Interpretation


<h6>Here are the actual results of a few of the features from this dataset. In comparison to the example above, it appears that the income column is far from the correct distribution while the others are relatively close.</h6>

<h6>The data, however, only has 4,196 rows and therefore isn't big enough to completly make any final judgement. I'm not positive as the the exact reason that the income is so far off other than the data not being selected completely randomly.</h6>

<br><br>

<img src="./images/FeatCorrGraph.PNG" alt="Income Dependent Graph">

<h6>Bank, comercial, luxury assets have strong correlation as well as income and loan amount</h6>
<h6>Loan term, cibil score, and number of dependents, however, do not correlate much with any other feature</h6>

<br><br>

#### Feature Distributions 

<img src="./images/DistributionGraphs.PNG">

<br><br>

<img src="./images/CibilStatusScatter.PNG">

<h6>Clearly, the bank sets a fairly hard limit with anyone below a 550 credit score, with the occasional exception.</h6>
<h6>In choosing this data, it seems as though it was first filtered due to its clear lack of outliers.</h6><h6>Because the graph is so dense, I used DAX to create a new column to limit the number of points by half.</h6>

```
    filtered_cibil = 
    SWITCH(
        TRUE(),
        MOD(loan_approval_dataset[loan_id], 2) = 0, BLANK(),
        loan_approval_dataset[ cibil_score]
    )
```  

<br><br>

#### Assests by Loan Term

<img src="./images/AssetLoanTerm.PNG">


<h6>Surprisingly, the luxury assets on average are by far the most expensive, followed by residential.</h6>
<h6>Additionally, it appears that the asset values don't change too much by the loan term.</h6>


<br><br>

<img src="./images/IncomeDependentGraph.PNG" alt="Income Dependent Graph">





# Data Analysis

* <a href="./analysis.ipynb">Analysis Code</a>


# Machine Learning
## Decision Tree

## Neural Network


## Acknowledgments