import pandas as pd
import numpy as np



class TelcoDataCleaner:
    NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

    def __init__(self, missing_strategy="zero"):
        """
        missing_strategy:
            "zero" -> fill missing TotalCharges with 0
            "drop" -> drop rows where TotalCharges is missing
        """
        if missing_strategy not in {"zero", "drop"}:
            raise ValueError("missing_strategy has to be 'zero' or 'drop'.")

        self.missing_strategy = missing_strategy
        self.df = None



    def load_data(self, file_path):
        """Read the CSV into a DataFrame."""
        self.df = pd.read_csv(file_path)

        print(f"Loaded {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

        return self.df



    def drop_useless_columns(self):
        """customerID is unique per row, so it just gets in the way."""
        if "customerID" in self.df.columns:
            self.df.drop(columns=["customerID"], inplace=True)
            print("Dropped customerID.")

        return self.df


    def fix_data_types(self):
        """TotalCharges should be numeric, and SeniorCitizen reads better as Yes/No."""
        if "TotalCharges" in self.df.columns:
            self.df["TotalCharges"] = pd.to_numeric(
                self.df["TotalCharges"].astype(str).str.strip(),
                errors="coerce",
            )

        if "SeniorCitizen" in self.df.columns:
            self.df["SeniorCitizen"] = self.df["SeniorCitizen"].map(
                {0: "No", 1: "Yes"}
            )

        print("Data types fixed.")
        return self.df


    def handle_missing_values(self):
        """Handle the blanks in TotalCharges based on the chosen strategy."""
        print("\nMissing values before we touch anything:")
        print(self.df.isnull().sum()[lambda x: x > 0])

        if "TotalCharges" in self.df.columns:
            if self.missing_strategy == "zero":
                self.df["TotalCharges"] = self.df["TotalCharges"].fillna(0)
            else:
                self.df.dropna(subset=["TotalCharges"], inplace=True)

        print("\nMissing values after handling:")
        print(self.df.isnull().sum()[lambda x: x > 0])

        return self.df


    def remove_duplicates(self):
        """Check for duplicate rows and get rid of them."""
        duplicates = self.df.duplicated().sum()
        print(f"Found {duplicates} duplicate rows.")

        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        print(f"Rows left after dropping duplicates: {len(self.df)}")
        return self.df


    def inspect_categorical_columns(self):
        """Print out the unique values in each categorical column."""
        cat_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns

        print("\nCategorical columns and their values:")

        for col in cat_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts(dropna=False))

        return list(cat_cols)


    def inspect_numeric_columns(self):
        """Describe the numeric columns and plot boxplots to spot outliers."""
        cols = [col for col in self.NUMERIC_COLUMNS if col in self.df.columns]

        print("\nNumeric summary stats:")
        print(self.df[cols].describe())

        for col in cols:
            plt.figure(figsize=(8, 3))
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot of {col}")
            plt.tight_layout()
            plt.show()

        return self.df[cols].describe()


    def check_class_balance(self):
        """How many customers churned vs. stayed?"""
        if "Churn" not in self.df.columns:
            raise KeyError("There's no 'Churn' column in this dataset.")

        counts = self.df["Churn"].value_counts(dropna=False)
        proportions = (
            self.df["Churn"].value_counts(normalize=True, dropna=False) * 100
        )

        result = pd.DataFrame({"Count": counts, "Percentage": proportions})

        print("\nChurn class balance:")
        print(result.round(2))

        return result


    def check_multicollinearity(self, plot=True, calculate_vif=False):
        """Look at correlations and optionally compute VIFs."""
        cols = [col for col in self.NUMERIC_COLUMNS if col in self.df.columns]

        corr = self.df[cols].corr()

        print("\nCorrelation matrix:")
        print(corr)

        if plot:
            plt.figure(figsize=(7, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.show()

        if calculate_vif:
            # VIF needs complete observations with finite values.
            X = self.df[cols].replace([np.inf, -np.inf], np.nan).dropna()

            # Constant columns break VIF, so drop them.
            X = X.loc[:, X.nunique() > 1]

            if X.shape[1] < 2:
                print("Not enough varying features to compute VIF.")
                return corr, None

            # Adding a constant gives VIF values that assume an intercept.
            from statsmodels.tools.tools import add_constant
            X = add_constant(X, has_constant="add")

            vif = pd.DataFrame({
                "Feature": X.columns,
                "VIF": [
                    variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])
                ],
            })

            print("\nVariance Inflation Factors:")
            print(vif)

            return corr, vif

        return corr


    def sanity_checks(self):
        """Make sure the numbers and categories actually make sense."""
        print("\nRunning sanity checks...")

        if {"tenure", "MonthlyCharges", "TotalCharges"}.issubset(
            self.df.columns
        ):
            zero_tenure = self.df[self.df["tenure"] == 0]

            print("\nCustomers with zero tenure:")
            print(
                zero_tenure[["tenure", "MonthlyCharges", "TotalCharges"]]
            )

            for col in ["MonthlyCharges", "TotalCharges"]:
                assert (self.df[col] >= 0).all(), (
                    f"Negative values snuck into {col}."
                )

            print("Charge checks passed.")

        if "gender" in self.df.columns:
            expected = {"Female", "Male"}
            actual = set(self.df["gender"].dropna().unique())

            print("\nGender categories found:", actual)

            unexpected = actual - expected
            if unexpected:
                print("Unexpected gender values:", unexpected)
            else:
                print("Gender categories look good.")

        return True
