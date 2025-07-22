import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df, title="Dataset"):
    """
    Performs enhanced exploratory data analysis (EDA) on a DataFrame.
    Includes:
    - Basic info (data types, non-null counts).
    - Descriptive statistics.
    - Class distribution plot if 'class' or 'Class' column exists.
    - Additional analyses based on dataset type (e-commerce or credit card).
    """
    print(f"\n--- EDA for {title} ---")
    print(f"\nInfo for {title}:")
    df.info()

    print(f"\nDescriptive Statistics for {title}:")
    print(df.describe())

    # Check for target variable and plot distribution
    target_col = None
    if 'class' in df.columns:
        target_col = 'class'
    elif 'Class' in df.columns:
        target_col = 'Class'

    if target_col:
        print(f"\nClass Distribution for {title} ({target_col}):")
        class_counts = df[target_col].value_counts(normalize=True)
        class_counts_raw = df[target_col].value_counts()
        print(f"Proportions: {class_counts}")
        print(f"Raw Counts: {class_counts_raw}")
        print(f"Imbalance Ratio (Non-Fraud/Fraud): {class_counts_raw[0] / class_counts_raw[1]:.2f}")

        plt.figure(figsize=(6, 4))
        sns.barplot(x=class_counts.index, y=class_counts.values, palette=['#1f77b4', '#2ca02c'])
        plt.title(f'Class Distribution for {title}')
        plt.xlabel('Class')
        plt.ylabel('Proportion')
        plt.xticks(ticks=[0, 1], labels=['Non-Fraud (0)', 'Fraud (1)'])
        plt.show()
    else:
        print(f"\nNo 'class' or 'Class' column found for distribution plot in {title}.")

    # Enhanced EDA based on dataset type
    if 'purchase_time' in df.columns:  # E-commerce fraud data
        print("\n--- Additional EDA for E-commerce Fraud Data ---")
        
        # Time-based analysis
        df['purchase_hour'] = df['purchase_time'].dt.hour
        plt.figure(figsize=(10, 4))
        sns.histplot(data=df, x='purchase_hour', hue=target_col, multiple='stack', palette=['#1f77b4', '#2ca02c'])
        plt.title(f'Purchase Hour Distribution by Class for {title}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Count')
        plt.show()

        # Age and Sex distribution
        plt.figure(figsize=(10, 4))
        sns.histplot(data=df, x='age', hue='sex', multiple='stack', palette=['#ff7f0e', '#d62728'])
        plt.title(f'Age Distribution by Sex for {title}')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.show()

        # Purchase value vs. fraud
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=target_col, y='purchase_value', data=df, palette=['#1f77b4', '#2ca02c'])
        plt.title(f'Purchase Value Distribution by Class for {title}')
        plt.xlabel('Class')
        plt.ylabel('Purchase Value ($)')
        plt.xticks(ticks=[0, 1], labels=['Non-Fraud (0)', 'Fraud (1)'])
        plt.show()

    elif 'Time' in df.columns:  # Credit card fraud data
        print("\n--- Additional EDA for Credit Card Fraud Data ---")
        
        # Time distribution (normalized to hours)
        df['Time_hours'] = df['Time'] / 3600  # Convert seconds to hours
        plt.figure(figsize=(10, 4))
        sns.histplot(data=df, x='Time_hours', hue='Class', multiple='stack', palette=['#1f77b4', '#2ca02c'])
        plt.title(f'Transaction Time Distribution by Class for {title}')
        plt.xlabel('Time (hours)')
        plt.ylabel('Count')
        plt.show()

        # Amount vs. fraud
        plt.figure(figsize=(10, 4))
        sns.boxplot(x='Class', y='Amount', data=df, palette=['#1f77b4', '#2ca02c'])
        plt.title(f'Transaction Amount Distribution by Class for {title}')
        plt.xlabel('Class')
        plt.ylabel('Amount ($)')
        plt.xticks(ticks=[0, 1], labels=['Non-Fraud (0)', 'Fraud (1)'])
        plt.show()

    else:
        print("\nNo specific columns found for enhanced EDA.")

