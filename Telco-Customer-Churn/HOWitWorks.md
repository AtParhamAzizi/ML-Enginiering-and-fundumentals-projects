# How Multiple Linear Regression Works

## 1. Understanding the Basic Idea

Multiple Linear Regression (MLR) is a machine learning algorithm used to predict a continuous numerical value based on two or more input variables.

The main idea is simple: the algorithm looks at existing data, learns how different variables relate to the target, and uses those relationships to make predictions about new data.

For example, imagine you want to predict a house's price. You could use features such as its size, number of bedrooms, location, and age. Multiple Linear Regression analyzes these features to estimate the house's price.

Unlike simple linear regression, which uses only one predictor, multiple linear regression considers several predictors at the same time.

## 2. The Mathematical Model

The general formula for Multiple Linear Regression is:

```text
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Each part of the equation has a specific meaning:

- **y:** The target value we want to predict.
- **β₀:** The intercept, representing the predicted value when all input features are zero.
- **β₁, β₂, ..., βₙ:** The coefficients that describe how each feature affects the prediction, assuming the other features remain constant.
- **x₁, x₂, ..., xₙ:** The input features used to make the prediction.
- **ε:** The error term, representing the part of the target that the model cannot explain.

For example, when predicting house prices, a coefficient associated with house size tells us how much the predicted price changes when the size increases by one unit, assuming the other features remain unchanged.

The model combines the contributions of all these features to produce its final prediction.

## 3. How Does the Model Learn?

During training, Multiple Linear Regression tries to find the best coefficients for the input features.

To do this, it compares its predictions with the actual values in the training dataset. The differences between these values are called residuals.

The algorithm aims to minimize the **Sum of Squared Errors (SSE)**:

```text
SSE = Σ (y_actual - y_predicted)²
```

Squaring the errors prevents positive and negative errors from canceling each other out and gives larger errors more influence on the final result.

This optimization method is known as **Ordinary Least Squares (OLS)**.

For the standard linear regression model, the coefficients can be expressed mathematically as:

```text
β = (XᵀX)⁻¹Xᵀy
```

Here:

- **X** is the feature matrix, including a column of ones for the intercept.
- **Xᵀ** is the transpose of the feature matrix.
- **y** contains the actual target values.
- **β** represents the coefficients the model needs to learn.

This formula assumes that the required inverse exists. In practice, libraries such as scikit-learn typically use numerically stable methods, such as least-squares solvers based on singular value decomposition, rather than directly calculating the inverse of \(X^TX\).

Once the coefficients have been learned, the model can use them to predict values for new observations.

## 4. Important Assumptions

Multiple Linear Regression works best when certain assumptions are reasonably satisfied. Understanding these assumptions helps us decide whether the model is appropriate for a particular dataset.

1. **Linearity:** The expected relationship between the predictors and the target is linear in the model's coefficients.

2. **Independence of observations:** The observations should be independent of one another. In regression inference, the errors are also typically assumed to be independent.

3. **Homoscedasticity:** The variance of the errors should remain approximately constant across different predicted values or levels of the predictors.

4. **Normality of errors:** For standard statistical inference, the errors are often assumed to be normally distributed, conditional on the predictors. Normality is not required simply to calculate the ordinary least-squares estimates.

5. **No perfect multicollinearity:** One predictor should not be an exact linear combination of the others. Strong correlations between predictors can also make individual coefficients difficult to interpret.

6. **No autocorrelation:** When observations have a natural order, such as time-series data, errors should not exhibit problematic correlations across observations.

These assumptions do not all serve the same purpose. Some affect the reliability of predictions, while others are particularly important when interpreting coefficients and performing statistical tests.

## 5. How Do We Evaluate the Model?

After training the model, we need to measure how well it performs. Several metrics can help us understand its strengths and weaknesses.

### R-squared (R²)

R-squared measures how much of the variation in the target variable is explained by the model relative to a baseline that predicts the target's mean.

A higher value often indicates a better fit on the evaluated dataset. However, a high R² does not necessarily mean that the model will perform well on unseen data.

### Adjusted R-squared

Adjusted R-squared accounts for the number of predictors in the model. Unlike ordinary R², it can decrease when additional features do not provide enough explanatory value.

It is especially useful when comparing regression models with different numbers of predictors on the same dataset.

### Root Mean Squared Error (RMSE)

RMSE measures the typical magnitude of prediction errors while giving larger errors more weight.

Because it uses the square root of the average squared errors, it is expressed in the same units as the target variable.

### Mean Absolute Error (MAE)

MAE calculates the average absolute difference between actual and predicted values.

It is straightforward to interpret and is less sensitive to large errors than RMSE.

### P-values and t-statistics

These statistics are commonly used to assess whether individual regression coefficients provide evidence of an association with the target, under the model's assumptions.

They are useful for statistical interpretation, but they do not directly measure predictive performance.

### Residual plots

Residual plots help us identify patterns that the model may have missed. For example, a curved pattern can indicate a nonlinear relationship, while a funnel-shaped pattern may suggest non-constant error variance.

Evaluating these plots alongside numerical metrics gives us a better understanding of whether the model is appropriate for the data.

## 6. An Important Consideration for the Churn Prediction Project

In this project, the target variable is **`Churn`**, which indicates whether a customer leaves the service. Its values are typically represented as `Yes` or `No`.

This makes churn prediction a **classification problem**, rather than a standard regression problem.

Multiple Linear Regression is designed to predict continuous numerical values. Applying it directly to a binary target can produce predictions outside the range of 0 to 1, making the results unsuitable as direct churn probabilities.

For this reason, a classification algorithm is a better choice.

Some suitable options include:

- **Logistic Regression:** A good baseline model that estimates the probability of a customer churning.
- **Random Forest Classifier:** An ensemble model that can learn nonlinear relationships and interactions between features.
- **XGBoost Classifier:** A gradient-boosting algorithm that often performs well on structured datasets.
- **Support Vector Machine (SVM):** A classification method that can learn decision boundaries between classes.
- **Naive Bayes:** A probabilistic classifier that can serve as a simple baseline in suitable settings.

For a first machine learning project, I would recommend starting with Logistic Regression. It is relatively easy to understand, provides interpretable results, and offers a useful baseline for comparison with more complex models.

Multiple Linear Regression would be more appropriate if the goal were to predict a continuous variable, such as `MonthlyCharges` or `TotalCharges`, using the other available features.

## 7. Conclusion

Multiple Linear Regression is an important machine learning algorithm because it provides a straightforward way to model relationships between several input features and a continuous target variable.

Learning how it works introduces several fundamental concepts, including model coefficients