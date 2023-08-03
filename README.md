

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

<img src="./images/BenfordExample.PNG" alt="Ideal Benford's Law Graph">

<img src="./images/BenfordActual.PNG" alt="Benford's Law on a few features">

##### Interpretation


<h6>Here are the actual results of a few of the features from this dataset. In comparison to the example above, it appears that the income column is far from the correct distribution while the others are relatively close.</h6>

<h6>The data, however, only has 4,196 rows and therefore isn't big enough to completly make any final judgement. I'm not positive as the the exact reason that the income is so far off other than the data not being selected completely randomly.</h6>

<br><br>

<img src="./images/FeatCorrGraph.PNG" alt="Feature Correlation Graph">

<h6>Bank, comercial, luxury assets have strong correlation as well as income and loan amount</h6>
<h6>Loan term, cibil score, and number of dependents, however, do not correlate much with any other feature</h6>

<br><br>

#### Feature Distributions 

<img src="./images/DistributionGraphs.PNG" alt="Distributions Graph">

<h6>I'm surprised to see hardly any sure normal distributions. Unfortunately, this may be because the dataset is so small. Additionally, because the annual income is not normally distributed (as it should be) and did not even come close to passing Benford's Law, I wonder how this dataset was selected. One guess I have was that somehow it was chosen based on income. The loan success rate in this data is about 62.22%.</h6>

<br><br>

<img src="./images/CibilStatusScatter.PNG" alt="Credit Score Scatter Plot">

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

<img src="./images/AssetLoanTerm.PNG" alt='Loan by Term Plot'>


<h6>Surprisingly, the luxury assets on average are by far the most expensive, followed by residential.</h6>
<h6>Additionally, it appears that the asset values don't change too much by the loan term.</h6>


<br><br>

<img src="./images/IncomeDependentGraph.PNG" alt="Income Dependent Graph">


# Data Analysis

* <a href="./analysis.ipynb">Analysis Code</a>

#### Data Cleaning

<h6>The dataset was very clean. This was unfortunate because I wanted some extra pratice working with more difficult data.</h6>

##### Missing Values

<h6>There were no missing values in the data, nor were there any incomplete entries in any of the categorical data.</h6>

##### Outliers

<h6>I looked into the continuous features for any values high or lower than three standard deviations from the average. There were 33 total that were above the upper limit and none below the lower. The values were split among the residential and comercial assets. I decided to take the values down to simply equal the upper limit.</h6>

##### Negative Values

<h6>There were 28 negative values in the data. All were from the residential assests column and were the exact same value. I thought this was odd and found that most of the instances had different values in the rest of their respective rows. I'm not positve what the cause is, but I just changed them all to zero.</h6>

#### Analysis & Feature Engineering

##### Total Collateral & Loan Ratio

<h6>I created a new feature that was the sum of all the four assets. I then created a new column that was the ratio of the loan amount to the total collateral. I thought this would be a good indicator of the risk of the loan and will help me with both the machine learning later and further analysis.</h6>

##### Loan Amount and Income Ratio

<h6>I made a new column by taking the loan amount and dividing it by the annual income. The average is about 2.88. I thought this would be a valuable new feature because at a glance I could see if the income was too small to be able to support the debt in the future.</h6>

##### Credit Score Odd Values

<h6>Looking at the credit score scatter plot found above, I decided to look into the few values that were above a 550 credit score but were still rejected. There were only 13 in total. The primary finding I found was that the collateral was low. The total collateral to loan ratio average for the 13 was 1.08 (meaning the collateral was the same or more than the loan) while the average ratio for the entire dataset was 0.489 (the collateral being about double the loan).</h6>

<h6>Additionally, I took a look at the proportion of the loan to the income. The mean for the entire dataset is 2.88 (the loan being almost three times the annual income). On the other hand, the mean for the 13 rejected loans was 3.66. This seems to be another strong indicator of whether a loan is approved or rejected because a rule of thumb is that the annual income should be roughly a third of the loan.</h6>

# Machine Learning

## Decision Tree

## Neural Network


## Acknowledgments