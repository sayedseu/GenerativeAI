import ollama
import sqlite3
from sql_metadata import Parser
import re
import pandas as pd
import time
from typing import List, Dict, Tuple


class TextToSQLEvaluator:
    def __init__(self, model1_name: str, model2_name: str, database_path: str = None):
        """
        Initialize the evaluator with two Ollama models and custom prompt template.

        Args:
            model1_name: Name of the first Ollama model
            model2_name: Name of the second Ollama model
            database_path: Path to SQLite database for validation (optional)
        """
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.database_path = database_path
        self.results = []

        # Database schema from your prompt
        self.schema = """CREATE TABLE products (
  product_id INTEGER PRIMARY KEY,
  name VARCHAR(50),
  price DECIMAL(10,2),
  quantity INTEGER
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY,
   name VARCHAR(50),
   address VARCHAR(100)
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY,
  name VARCHAR(50),
  region VARCHAR(50)
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY,
  product_id INTEGER,
  customer_id INTEGER,
  salesperson_id INTEGER,
  sale_date DATE,
  quantity INTEGER
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY,
  product_id INTEGER,
  supply_price DECIMAL(10,2)
);"""

        # Test Ollama connection
        try:
            ollama.list()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")

        # Test database connection if provided
        if database_path:
            try:
                self.conn = sqlite3.connect(database_path)
                self.cursor = self.conn.cursor()
            except Exception as e:
                raise ConnectionError(f"Failed to connect to database: {e}")

    def generate_sql(self, question: str, model_name: str) -> str:
        """
        Generate SQL using your exact prompt structure.

        Args:
            question: Natural language question
            model_name: Which model to use

        Returns:
            Generated SQL query
        """
        prompt = f"""### Instructions:
Your task is to convert a question into a SQL query, given a Postgres database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{question}`.
This query will run on a database whose schema is represented in this string:
{self.schema}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
```sql
"""
        try:
            response = ollama.generate(
                model=model_name,
                prompt=prompt,
                stream=False
            )
            # Extract SQL from between ```sql ``` markers
            sql = response['response'].strip()
            match = re.search(r'```sql\n(.*?)\n```', sql, re.DOTALL)
            return match.group(1).strip() if match else sql
        except Exception as e:
            print(f"Error generating SQL with {model_name}: {e}")
            return ""

    def validate_sql(self, query: str) -> dict:
        """
        Comprehensive SQL validation with your specific requirements.

        Returns dictionary with:
        - valid_syntax: Basic SQL syntax check
        - valid_execution: Runs against database (if available)
        - uses_aliases: Checks for table alias usage
        - proper_float_casting: Checks ratio float casting
        - required_tables: Tables that should be included
        - required_columns: Columns that should be included
        """
        if not query:
            return {
                'valid_syntax': False,
                'valid_execution': False,
                'uses_aliases': False,
                'proper_float_casting': False,
                'required_tables': [],
                'required_columns': []
            }

        validation = {
            'valid_syntax': False,
            'valid_execution': False,
            'uses_aliases': False,
            'proper_float_casting': False
        }

        # Basic syntax validation
        try:
            parser = Parser(query)
            validation['valid_syntax'] = True
            validation['required_tables'] = parser.tables
            validation['required_columns'] = parser.columns
        except:
            return validation

        # Execution validation (if database available)
        if self.database_path:
            try:
                if not any(keyword in query.upper() for keyword in ['DROP', 'DELETE', 'TRUNCATE', 'UPDATE']):
                    self.cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                    validation['valid_execution'] = True
            except:
                pass

        # Check for table aliases usage
        from_match = re.search(r'FROM\s+([^\s,(]+)(?:\s+AS\s+|\s+)([^\s,)]+)', query, re.IGNORECASE)
        join_matches = re.findall(r'JOIN\s+([^\s,(]+)(?:\s+AS\s+|\s+)([^\s,)]+)', query, re.IGNORECASE)
        column_refs = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\b', query)

        validation['uses_aliases'] = bool(from_match) and (bool(join_matches) or not join_matches) and bool(column_refs)

        # Check for proper float casting in ratios
        validation['proper_float_casting'] = bool(
            re.search(r'CAST\(.*?AS\s+FLOAT\)\s*\/', query, re.IGNORECASE) or
            re.search(r'\/.*?CAST\(.*?AS\s+FLOAT\)', query, re.IGNORECASE) or
            not re.search(r'\/', query)  # No division found
        )

        return validation

    def evaluate_question(self, question: str, question_type: str) -> dict:
        """
        Full evaluation of both models on a single question.
        """
        # Generate SQL from both models
        start_time1 = time.time()
        sql1 = self.generate_sql(question, self.model1_name)
        time1 = time.time() - start_time1

        start_time2 = time.time()
        sql2 = self.generate_sql(question, self.model2_name)
        time2 = time.time() - start_time2

        # Validate both queries
        validation1 = self.validate_sql(sql1)
        validation2 = self.validate_sql(sql2)

        # Prepare result
        result = {
            'question': question,
            'question_type': question_type,
            'model1': self.model1_name,
            'model2': self.model2_name,
            'sql1': sql1,
            'sql2': sql2,
            'time1': time1,
            'time2': time2,
            **{f'model1_{k}': v for k, v in validation1.items()},
            **{f'model2_{k}': v for k, v in validation2.items()}
        }

        self.results.append(result)
        return result

    def analyze_results(self) -> dict:
        """
        Comprehensive analysis of all evaluation results.
        """
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        # Basic metrics
        metrics = {
            'total_questions': len(df),
            'questions_by_type': dict(df['question_type'].value_counts()),

            # Model 1 metrics
            'model1_syntax_accuracy': df['model1_valid_syntax'].mean(),
            'model1_execution_accuracy': df['model1_valid_execution'].mean(),
            'model1_alias_usage': df['model1_uses_aliases'].mean(),
            'model1_float_casting': df['model1_proper_float_casting'].mean(),
            'model1_avg_time': df['time1'].mean(),

            # Model 2 metrics
            'model2_syntax_accuracy': df['model2_valid_syntax'].mean(),
            'model2_execution_accuracy': df['model2_valid_execution'].mean(),
            'model2_alias_usage': df['model2_uses_aliases'].mean(),
            'model2_float_casting': df['model2_proper_float_casting'].mean(),
            'model2_avg_time': df['time2'].mean(),

            # Comparison metrics
            'syntax_agreement': (df['model1_valid_syntax'] == df['model2_valid_syntax']).mean(),
            'execution_agreement': (df['model1_valid_execution'] == df['model2_valid_execution']).mean()
        }

        # Metrics by question type
        for q_type in df['question_type'].unique():
            type_df = df[df['question_type'] == q_type]
            metrics.update({
                f'model1_syntax_{q_type}': type_df['model1_valid_syntax'].mean(),
                f'model2_syntax_{q_type}': type_df['model2_valid_syntax'].mean(),
                f'model1_execution_{q_type}': type_df['model1_valid_execution'].mean(),
                f'model2_execution_{q_type}': type_df['model2_valid_execution'].mean()
            })

        return metrics

    def get_recommendation(self) -> dict:
        """
        Data-driven recommendation with scoring based on your requirements.
        """
        metrics = self.analyze_results()

        if not metrics:
            return {'recommendation': 'Insufficient data', 'reason': 'No evaluations performed'}

        # Weighted scoring (adjust weights as needed)
        weights = {
            'syntax': 0.3,
            'execution': 0.3,
            'aliases': 0.2,
            'float_casting': 0.1,
            'speed': 0.1
        }

        # Calculate scores
        model1_score = (
                weights['syntax'] * metrics['model1_syntax_accuracy'] +
                weights['execution'] * metrics['model1_execution_accuracy'] +
                weights['aliases'] * metrics['model1_alias_usage'] +
                weights['float_casting'] * metrics['model1_float_casting'] +
                weights['speed'] * (
                            1 - metrics['model1_avg_time'] / max(metrics['model1_avg_time'], metrics['model2_avg_time'],
                                                                 1))
        )

        model2_score = (
                weights['syntax'] * metrics['model2_syntax_accuracy'] +
                weights['execution'] * metrics['model2_execution_accuracy'] +
                weights['aliases'] * metrics['model2_alias_usage'] +
                weights['float_casting'] * metrics['model2_float_casting'] +
                weights['speed'] * (
                            1 - metrics['model2_avg_time'] / max(metrics['model1_avg_time'], metrics['model2_avg_time'],
                                                                 1))
        )

        # Prepare recommendation
        recommendation = {
            'recommended_model': self.model1_name if model1_score > model2_score else self.model2_name,
            'model1_score': round(model1_score, 3),
            'model2_score': round(model2_score, 3),
            'score_difference': round(abs(model1_score - model2_score), 3),
            'strengths': {
                self.model1_name: [],
                self.model2_name: []
            }
        }

        # Identify strengths
        if metrics['model1_syntax_accuracy'] > metrics['model2_syntax_accuracy']:
            recommendation['strengths'][self.model1_name].append(
                f"Better syntax accuracy ({metrics['model1_syntax_accuracy']:.2f} vs {metrics['model2_syntax_accuracy']:.2f})")
        else:
            recommendation['strengths'][self.model2_name].append(
                f"Better syntax accuracy ({metrics['model2_syntax_accuracy']:.2f} vs {metrics['model1_syntax_accuracy']:.2f})")

        if metrics['model1_alias_usage'] > metrics['model2_alias_usage']:
            recommendation['strengths'][self.model1_name].append(
                f"Better alias usage ({metrics['model1_alias_usage']:.2f} vs {metrics['model2_alias_usage']:.2f})")
        else:
            recommendation['strengths'][self.model2_name].append(
                f"Better alias usage ({metrics['model2_alias_usage']:.2f} vs {metrics['model1_alias_usage']:.2f})")

        # Add type-specific strengths
        for q_type in set([r['question_type'] for r in self.results]):
            m1_acc = metrics.get(f'model1_syntax_{q_type}', 0)
            m2_acc = metrics.get(f'model2_syntax_{q_type}', 0)

            if m1_acc > m2_acc:
                recommendation['strengths'][self.model1_name].append(
                    f"Better at {q_type} questions ({m1_acc:.2f} vs {m2_acc:.2f})")
            elif m2_acc > m1_acc:
                recommendation['strengths'][self.model2_name].append(
                    f"Better at {q_type} questions ({m2_acc:.2f} vs {m1_acc:.2f})")

        return recommendation

    def save_results(self, filename: str = 'sql_generation_evaluation.csv'):
        """Save detailed results to CSV."""
        pd.DataFrame(self.results).to_csv(filename, index=False)


# Example Usage
if __name__ == "__main__":
    # Initialize with your preferred models
    evaluator = TextToSQLEvaluator(
        model1_name="deepseek-coder",
        model2_name="llama3.2",
        database_path="sales_db.db"  # Optional
    )

    # Define your test questions
    test_questions = [
        ("List all products priced above $100", "simple"),
        ("Show total sales by region", "aggregate"),
        ("Find customers who bought more than 5 items in January", "temporal"),
        ("What is the average price of products sold by each salesperson?", "aggregate_join"),
        ("Calculate the profit margin (price - supply_price)/price for each product", "ratio"),
        ("Which products have never been sold?", "complex"),
        ("List salespeople with their total sales amount, ordered by highest first", "ranking"),
        ("What percentage of products are low stock (quantity < 20)?", "percentage")
    ]

    # Run evaluations
    for question, q_type in test_questions:
        evaluator.evaluate_question(question, q_type)
        print(f"Evaluated: {question}")

    # Get results
    metrics = evaluator.analyze_results()
    recommendation = evaluator.get_recommendation()

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Models compared: {evaluator.model1_name} vs {evaluator.model2_name}")
    print(f"Total questions evaluated: {metrics['total_questions']}")
    print(f"\nSyntax Accuracy: {metrics['model1_syntax_accuracy']:.2f} vs {metrics['model2_syntax_accuracy']:.2f}")
    print(
        f"Execution Accuracy: {metrics['model1_execution_accuracy']:.2f} vs {metrics['model2_execution_accuracy']:.2f}")
    print(f"Alias Usage: {metrics['model1_alias_usage']:.2f} vs {metrics['model2_alias_usage']:.2f}")
    print(f"Float Casting: {metrics['model1_float_casting']:.2f} vs {metrics['model2_float_casting']:.2f}")
    print(f"Average Time: {metrics['model1_avg_time']:.2f}s vs {metrics['model2_avg_time']:.2f}s")

    print("\n=== Recommendation ===")
    print(f"Recommended model: {recommendation['recommended_model']}")
    print(f"Score: {recommendation['model1_score']} vs {recommendation['model2_score']}")
    print("\nStrengths:")
    for model, strengths in recommendation['strengths'].items():
        print(f"- {model}:")
        for strength in strengths:
            print(f"  • {strength}")

    # Save detailed results
    evaluator.save_results()
    print("\nDetailed results saved to 'sql_generation_evaluation.csv'")